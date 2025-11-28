from typing import List, Dict
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data


class JMDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        collate_fn = kwargs.get("collate_fn", collate_pairwise)
        kwargs = {k: v for k, v in kwargs.items() if k != "collate_fn"}
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)

class JMTopoXDataLoader(DataLoader):
    """DataLoader for pairwise data with TopoX-style key mapping."""
    def __init__(self, dataset, **kwargs):
        collate_fn = kwargs.get("collate_fn", collate_pairwise_topox)
        kwargs = {k: v for k, v in kwargs.items() if k != "collate_fn"}
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)

def collate_pairwise(batch: List[Data]):
    connectivity_types = [
        'incidence',
        'adjacency',
        'coadjacency'
        #Todo: add the laplacian and hodge laplacian
    ]
    batched_data = Data()
    #Todo: @Martin is this correct - since we flatten the pairwise data object?
    batched_data.batch_size = len(batch) * 2

    # Get the dimension of the triangulation (both triangulations have the same dimension)
    max_dimension = int(batch[0]["triangulation_1"]["dimension"].item())

    # Get the connectivity matrix names
    connectivity_matrices = [k for k in batch[0]["triangulation_1"]['connectivity'].keys() if k.split('_')[0] in connectivity_types]
    print([k for k in batch[0]["triangulation_1"]['connectivity'].keys()])

    batched_list_connectivities: Dict[str, List] = defaultdict(list)
    batched_features: Dict[int, List] = defaultdict(list)
    batched_ys: List = []

    running_idx_r_simplex = defaultdict(int)

    # For each pair of triangulation that we want to `flatten` while collating
    for pair in batch:
        batched_ys.append(pair.y)
        t_1, t_2 = pair["triangulation_1"], pair["triangulation_2"]

        # For each of the two triangulations
        for t in [t_1, t_2]:
            cnt_n_simplices: Dict = defaultdict(int)

            # For every dimension
            for dimension in range(0, max_dimension+1):
                cnt_n_simplices[dimension] = t[f'num_{dimension}_simplices'] 

            # Do batching for connetivity
            for connect_mat in connectivity_matrices:
                t['connectivity'][connect_mat] =  t['connectivity'][connect_mat].to_dense().nonzero()
                if 'incidence' in connect_mat:
                    rank_to = int(connect_mat.split('_')[1]) 
                    rank_from = rank_to - 1

                    # TODO: Check direction
                    r_from_tensor = t['connectivity'][connect_mat][0] 
                    r_from_tensor += running_idx_r_simplex[rank_from]
                    running_idx_r_simplex[rank_from] += cnt_n_simplices[rank_from]

                    r_to_tensor = t['connectivity'][connect_mat][1] 
                    r_to_tensor += running_idx_r_simplex[rank_to]
                    running_idx_r_simplex[rank_to] += cnt_n_simplices[rank_to]

                    # Concat both tensor again
                    r_final_tensor = torch.cat([r_from_tensor, r_to_tensor], dim=0)
                    batched_list_connectivities[connect_mat].append(r_final_tensor)
                    
                elif 'laplacian' in connect_mat:
                    raise NotImplementedError("TODO")
                # TODO:  Check if in does substrings
                elif 'adjacency' in connect_mat or 'coadjacency' in connect_mat:
                    rank = int(connect_mat.split('_')[1])

                    # Update with off
                    t['connectivity'][connect_mat] += running_idx_r_simplex[rank]

                    # Update the offset
                    running_idx_r_simplex[rank] += cnt_n_simplices[rank]

                    batched_list_connectivities[connect_mat].append(t['connectivity'][connect_mat])
                else:
                    raise KeyError(f"Invalid connectivity matrix {connect_mat}")
           # Do batching for features         
            for dimension in range(0, max_dimension+1):
                batched_features[dimension].append(t["x"][dimension])


    # Initialize tensor x
    batched_data["x"] = dict()
    x_belonging = dict()

    # Concat the tensors for the batch
    for dimension in range(0, max_dimension+1):
        batched_data["x"][dimension] = torch.cat(batched_features[dimension], dim=0)
        signals_of_dim_belonging = [
            torch.tensor([i] * len(batch[i].x[dimension]), dtype=torch.int64)
            for i in range(len(batch))
            if dimension in batch[i].x
        ]
        x_belonging[dimension] = torch.cat(signals_of_dim_belonging, dim=0)

    for connect_mat in connectivity_matrices:
        batched_data[connect_mat] = torch.cat(batched_list_connectivities[connect_mat], dim=0)

    batched_data.y = torch.cat(batched_ys, dim=0)

    return batched_data

def collate_pairwise_topox(batch: List[Data]):
    batched_data = Data()
    batched_data.batch_size = len(batch) * 2
    collated_batch = collate_pairwise(batch)
    batched_data.x_belonging = collated_batch.x_belonging
    batched_data.x_batched = collated_batch.x

    #Todo: Add correct connectivity matrices here
    batched_data.connectivity = collated_batch.connectivity


    batched_data.y = collated_batch.y

    return batched_data



def collate_simplicial_models_topox(batch):
    batched_data = Data()
    batched_data.batch_size = len(batch)
    # First, batch signals
    x_batched, x_belonging = collate_signals(batch)
    batched_data.x = x_batched
    batched_data.x_belonging = x_belonging
    # Second, batch structure matrices
    batched_data.connectivity = collate_connectivity_matrices(batch)
    # Third, batch output
    if len(batch[0].y.shape) == 0:
        # We directly create the tensor because we have scalars.
        batched_data.y = torch.tensor([example.y for example in batch])
    else:
        # We concatenate the tensors.
        batched_data.y = torch.cat([example.y for example in batch], dim=0)
    return batched_data

