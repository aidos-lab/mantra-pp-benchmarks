from torch_geometric.utils import scatter

def mean_pool(x, batch):
    return scatter(x, batch, dim=-2, reduce='mean')