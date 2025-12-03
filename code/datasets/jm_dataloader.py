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

def collate_pairwise(batch: List[Data]):
    connectivity_types = [
        'incidence',
        'adjacency',
        'coadjacency',
        'laplacian',
    ]

    batched_data = Data()

    #Todo: @Martin is this correct - since we flatten the pairwise data object?
    #batched_data.batch_size = len(batch) * 2

    # Get the dimension of the triangulation (both triangulations have the same dimension)
    max_dimension = int(batch[0]["triangulation_1"]["dimension"].item())

    connectivity_matrices = []

    # Get the connectivity matrix names
    for k in batch[0]["triangulation_1"]['connectivity'].keys():
        split_k = k.split('_')
        if split_k[0] in connectivity_types or split_k[1] in connectivity_types:
            connectivity_matrices.append(k)


    batched_list_connectivities: Dict[str, List] = defaultdict(list) # Batching of connectivities
    batched_features: Dict[int, List] = defaultdict(list) # Batching of x_i
    batched_ys: List = [] # Batching of y
    batched_batch: Dict = defaultdict(list) # Batching of batch_i

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
                cnt_n_simplices[dimension] =  t['x'][dimension].shape[0]

           # Do batching for features         
            for dimension in range(0, max_dimension+1):
                batched_features[dimension].append(t["x"][dimension])
                batched_batch[dimension].append(torch.tensor([running_idx_r_simplex[dimension]] * cnt_n_simplices[dimension]))

            # Do batching for connetivity
            for connect_mat in connectivity_matrices:
                t['connectivity'][connect_mat] =  t['connectivity'][connect_mat].to_dense().nonzero().T
                if 'incidence' in connect_mat:
                    rank_to = int(connect_mat.split('_')[-1]) 
                    rank_from = rank_to - 1

                    # TODO: Check direction
                    r_from_tensor = t['connectivity'][connect_mat][0] 
                    r_from_tensor += running_idx_r_simplex[rank_from]

                    r_to_tensor = t['connectivity'][connect_mat][1] 
                    r_to_tensor += running_idx_r_simplex[rank_to]

                    # Concat both tensor again
                    r_final_tensor = torch.cat([r_from_tensor.unsqueeze(0), r_to_tensor.unsqueeze(0)], dim=0)
                    batched_list_connectivities[connect_mat].append(r_final_tensor)
                # TODO:  Check if in does substrings
                elif 'adjacency' in connect_mat or 'coadjacency' in connect_mat or 'down_laplacian' in connect_mat or 'up_laplacian' in connect_mat or 'hodge_laplacian' in connect_mat:
                    rank = int(connect_mat.split('_')[-1])

                    # Update with off
                    t['connectivity'][connect_mat] += running_idx_r_simplex[rank]

                    # Update the offset

                    batched_list_connectivities[connect_mat].append(t['connectivity'][connect_mat])
                else:
                    raise KeyError(f"Invalid connectivity matrix {connect_mat}")
            # Update running index
            for dimension in range(0, max_dimension+1):
                running_idx_r_simplex[dimension] += cnt_n_simplices[dimension]



    # Initialize connectivity
    batched_data['connectivity'] = defaultdict()

    # Set x-s as a tuple
    batched_data["x"] = [
        torch.cat(batched_features[dimension], dim=0)
        for dimension in range(0, max_dimension+1)
    ]
    
    # Set belonging as a tuple
    batched_data["x_belonging"] = [
        torch.cat(batched_batch[dimension], dim=0)
        for dimension in range(0, max_dimension+1)
    ]

    print(batched_data["x_belonging"][0].shape)


    # Connectivity matrices at least can be referenced as dictionaries ....
    for connect_mat in connectivity_matrices:
        batched_data['connectivity'][connect_mat] = torch.cat(batched_list_connectivities[connect_mat], dim=1)

    batched_data.y = torch.cat(batched_ys, dim=0)

    return batched_data
