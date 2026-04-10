import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from collections import defaultdict
import numpy as np

from typing import List, Dict, Any


# class SimplicialData(Data):
#     def __inc__(self, key, value, *args, **kwargs):
#         if any([con in key for con in ["laplacian", "adjacency", "hodge"]]):
#             rank = int(key.split('_')[-1])
#             return getattr(self, f"x_{rank}").size(0)
#         return super().__init__(key, value, *args, **kwargs)
#
#     def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
#         return super().__init__(key, value, *args, **kwargs)



def collate_simplicial(batch):
    VALID_CONNECTIVITY = [
            "laplacian",
            "adjacency",
            "hodge",
            "incidence"
    ]

    batched_data = Data()

    batched_data.b_size = len(batch)
    
    idx_batched = defaultdict(list)
    x_batched = defaultdict(list)
    y_batched = []
    connectivity_indices_batched: Dict[str, List] = defaultdict(list)
    connectivity_values_batched: Dict[str, List] = defaultdict(list)
    connectivity_shape: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    # Maximum simplex dimension
    max_dimension = batch[0].dimension.item()

    # Get the names of the connectivity matrices
    connectivity_types = []
    for k in batch[0].keys():
        if k == 'incidence_0': #TODO: Fix this case, but it's actually useless
            continue
        split_k = k.split("_")
        if split_k[0] in VALID_CONNECTIVITY or (len(split_k) > 1 and split_k[1] in VALID_CONNECTIVITY):
            connectivity_types.append(k)

    # Running index of each r-dimensional simplex
    running_idx_r_simplex = defaultdict(int)

    # For each simplicial complex to batch
    for i, sc in enumerate(batch):

        # 1. For every dimension batch the features
        for dim in range(max_dimension+1):
            x_feat = getattr(sc, f"x_{dim}")
            if isinstance(x_feat, np.ndarray):
                x_feat = torch.from_numpy(x_feat).to(dtype=torch.float32)
            x_batched[dim].append(x_feat)
            # Add the running index of the r-simplex (with r=dim) times the number of r-simplices that exist
            idx_batched[dim].append(torch.tensor([i] * getattr(sc, f"x_{dim}").shape[0]))

        # 2. Update the connectivity matrices
        for con_type in connectivity_types:
            A = sc[con_type]

            if A.is_sparse:
                A = A.coalesce()
                indices = A.indices()
                values = A.values()
            else:
                dense = A.to_dense()
                indices = A.nonzero(as_tuple=False).T
                values = dense[indices[0], indices[1]]

            # We are going to be modifing these
            indices = indices.clone()

            if "incidence" in con_type:
                # Extrat the ranks
                r_to = int(con_type.split("_")[-1])
                r_from = r_to - 1

                # Update indices for r_from
                indices[0] += running_idx_r_simplex[r_from]

                # Update indices for r_to
                indices[1] += running_idx_r_simplex[r_to]


                connectivity_shape[con_type][0] += getattr(sc, f"x_{r_from}").shape[0]
                connectivity_shape[con_type][1] += getattr(sc, f"x_{r_to}").shape[0]

            elif any([con in con_type for con in ['adjacency', 'coadjacency', 'down_laplacian', 'up_laplacian', 'hodge_laplacian']]):
                r = int(con_type.split("_")[-1])

                indices += running_idx_r_simplex[r]
                n = getattr(sc, f"x_{r}").shape[0]

                connectivity_shape[con_type][0] += n
                connectivity_shape[con_type][1] += n
            else:
                raise KeyError("Invalid connection mat", con_type)
            
            connectivity_indices_batched[con_type].append(indices)
            connectivity_values_batched[con_type].append(values)


        # 3. Batch the labels
        y_batched.append(sc.y)

        # Update the running idx
        for dim in range(max_dimension+1):
            running_idx_r_simplex[dim] += getattr(sc, f"x_{dim}").shape[0]


    # Assign to new data object x and batch idx
    for dim in range(max_dimension + 1):
        setattr(batched_data, f"x_{dim}", torch.cat(x_batched[dim], dim=0))
        setattr(batched_data, f"batch_{dim}", torch.cat(idx_batched[dim], dim=0))

    setattr(batched_data, "y", torch.cat(y_batched, dim=0))

    for con_type in connectivity_types:
        cat_indices = torch.cat(connectivity_indices_batched[con_type], dim=1)
        cat_values = torch.cat(connectivity_values_batched[con_type], dim=0)
        shape = connectivity_shape[con_type]
        sparse_mat = torch.sparse_coo_tensor(cat_indices, cat_values, size=shape).coalesce()

        setattr(batched_data, con_type, sparse_mat)

    return batched_data
