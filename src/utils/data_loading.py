import torch
from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import is_sparse

from collections import defaultdict
import numpy as np

from typing import List, Dict, Any
from torch_sparse import SparseTensor


# NOTE: Could be useful in the future ?
class SimplicialData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key.endswith('_idx'):
            prefix = key[:-4]
            if "incidence" in prefix:
                r_to = int(prefix.split("_")[-1])
                r_from = r_to - 1
                if r_from < 0:
                    return 0
            else:
                r_from = r_to = int(prefix.split("_")[-1])

            return torch.tensor([
                [self[f"x_{r_from}"].size(0)],
                [self[f"x_{r_to}"].size(0)]
            ])
        if key.startswith("batch_"):
            return 1
        return 0

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key.endswith("_idx"):
            return 1
        if key.endswith('_val'):
            return 0
        return 0

# class SimplicialData(Data):
#     def is_valid(self, string):
#         valid_names = ["adj", "incidence", "laplacian"]
#         return any(name in string for name in valid_names)

#     def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
#         if torch_geometric.utils.is_sparse(value) and self.is_valid(key):
#             return (0, 1)
#         elif "index" in key or key == "face":
#             return -1
#         else:
#             return 0


def convert_connectivity(A, key, data):
    if is_sparse(A):
        if isinstance(A, SparseTensor):
            row, col, val = A.coo()
            idx = torch.stack([row, col], dim=0)
        elif A.is_sparse:
            if not A.is_coalesced():
                A.coalesce()
            idx = A.indices()
            val = A.values()
    else:
        idx = A.nonzero(as_tuple=False).T
        val = A[idx[0], idx[1]]

    setattr(data, key+"_idx", idx.clone()) 
    setattr(data, key+"_val", val.float()) 

def collate_simplicial(batch):
    VALID_CONNECTIVITY = [
            "laplacian",
            "adjacency",
            "hodge",
            "incidence"
    ]

    # Max dim
    max_dimension = batch[0].dimension.item()

    # Calculate connectivity types
    connectivity_types = []
    for k in batch[0].keys():
        if k == 'incidence_0': #TODO: Fix this case, but it's actually useless
            continue
        split_k = k.split("_")
        if split_k[0] in VALID_CONNECTIVITY or (len(split_k) > 1 and split_k[1] in VALID_CONNECTIVITY):
            connectivity_types.append(k)

    data_list = []
    idx_batched = defaultdict(list)

    for i, sc in enumerate(batch):
        data = SimplicialData()
        for key in sc.keys():
            value = sc[key]

            if key in connectivity_types:
                convert_connectivity(value, key, data)
            else:
                if key == 'y':
                    assert isinstance(sc[key], torch.Tensor), "`y` is not a Tensor"
                data[key] = value

        # Fill in simplex counts
        simplex_cnts = {
            dim: data[f"x_{dim}"].shape[0]
            for dim in range(max_dimension + 1)
        }
        # Make batch dimension tensors
        for dim in range(max_dimension+1):
            # Add the running index of the r-simplex (with r=dim) times the number of r-simplices that exist
            idx_batched[dim].append(torch.full((simplex_cnts[dim],), i, dtype=torch.long))

        data_list.append(data)

    # for key in data_list[0].keys():
    #     val = data_list[0][key]
    #     result = data_list[0].__inc__(key, val)
    #     if result is None:
    #         print(f"None inc for key: {key}, type: {type(val)}, shape: {getattr(val, 'shape', 'N/A')}")

    batch = torch_geometric.data.Batch.from_data_list(data_list, exclude_keys=['simplex_trie', 'id', 'name'])

    for key, value in idx_batched.items():
        batch[f"batch_{key}"] = torch.cat(value, dim=0).long()


    if (batch.get("batch") is not None) and (batch.get("batch_0") is not None):
        # Back compatiility check
        assert torch.all(batch["batch_0"].cpu() == batch.pop("batch").cpu()), (
            "batch['batch_0'] and batch['batch'] should match in the number of nodes"
        )

    if (batch.get("batch") is not None) and (batch.get("batch_0") is None):
        # Back compatiility check
        # Instead of batch.batch as in the original code, we use batch.batch_0 always to refer to nodes
        batch["batch_0"] = batch.pop("batch")


    return batch



# Deprecated
def collate_simplicial_prev(batch):
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

    # Metadata to skip computing splits every loop
    con_meta = {}
    for con_type in connectivity_types:
        if "incidence" in con_type:
            r_to = int(con_type.split("_")[-1])
            con_meta[con_type] = ("rect", r_to - 1, r_to)
        elif any(con in con_type for con in ['adjacency', 'coadjacency', 'down_laplacian', 'up_laplacian', 'hodge_laplacian']):
            r = int(con_type.split("_")[-1])
            con_meta[con_type] = ("square", r, r)

    # Running index of each r-dimensional simplex
    running_idx_r_simplex = defaultdict(int)


    # Preallocating tensors
    total_nnz = defaultdict(int)
    for sc in batch:
        for con_type in connectivity_types:
            total_nnz[con_type] += sc[con_type]._nnz()

    idx_buffers = {ct: torch.empty((2, total_nnz[ct]), dtype=torch.long) for ct in connectivity_types}
    val_buffers = {ct: torch.empty(total_nnz[ct], dtype=torch.float32) for ct in connectivity_types}
    buf_offset = defaultdict(int)


    # For each simplicial complex to batch
    for i, sc in enumerate(batch):
        simplex_cnts = {dim: getattr(sc, f"x_{dim}").shape[0] for dim in range(max_dimension+1)}

        # 1. For every dimension batch the features
        for dim in range(max_dimension+1):
            x_feat = getattr(sc, f"x_{dim}")
            if isinstance(x_feat, np.ndarray):
                x_feat = torch.from_numpy(x_feat).to(dtype=torch.float32)
            x_batched[dim].append(x_feat)
            # Add the running index of the r-simplex (with r=dim) times the number of r-simplices that exist
            idx_batched[dim].append(torch.full((simplex_cnts[dim],), i, dtype=torch.long))

        # 2. Update the connectivity matrices
        for con_type in connectivity_types:
            A = sc[con_type]

            if A.is_sparse:
                if not A.is_coalesced():
                    A = A.coalesce()
                #A = SparseTensor.from_torch_sparse_coo_tensor()
                indices = A.indices().clone()
                values = A.values()
            else:
                print("BAD NOT SPARSE")
                indices = A.nonzero(as_tuple=False).T
                values = A[indices[0], indices[1]]

            # We are going to be modifing these
            mat_type, r_from, r_to = con_meta[con_type]

            if mat_type == 'rect':
                # Extrat the ranks

                # Update indices for r_from
                indices[0] += running_idx_r_simplex[r_from]

                # Update indices for r_to
                indices[1] += running_idx_r_simplex[r_to]


                #connectivity_shape[con_type][0] += simplex_cnts[r_from] 
                #connectivity_shape[con_type][1] += simplex_cnts[r_to]

            elif mat_type == 'square':
                indices += running_idx_r_simplex[r_to]
                #n = simplex_cnts[r]

                #connectivity_shape[con_type][0] += n
                #connectivity_shape[con_type][1] += n
            else:
                raise KeyError("Invalid connection mat", con_type)
            
            nnz = indices.shape[1]
            idx_buffers[con_type][:, buf_offset[con_type]:buf_offset[con_type] + nnz] = indices
            val_buffers[con_type][buf_offset[con_type]:buf_offset[con_type] + nnz] = values
            buf_offset[con_type] += nnz


        # 3. Batch the labels
        y_batched.append(sc.y)

        # Update the running idx
        for dim in range(max_dimension+1):
            running_idx_r_simplex[dim] += simplex_cnts[dim]


    # Assign to new data object x and batch idx
    for dim in range(max_dimension + 1):
        setattr(batched_data, f"x_{dim}", torch.cat(x_batched[dim], dim=0))
        setattr(batched_data, f"batch_{dim}", torch.cat(idx_batched[dim], dim=0))

    setattr(batched_data, "y", torch.cat(y_batched, dim=0))

    for con_type in connectivity_types:
        _, r_from, r_to = con_meta[con_type]
        #shape = connectivity_shape[con_type]
        sparse_mat = torch.sparse_coo_tensor(idx_buffers[con_type], val_buffers[con_type], size=(running_idx_r_simplex[r_from], running_idx_r_simplex[r_to])).coalesce()

        setattr(batched_data, con_type, sparse_mat)

    return batched_data
