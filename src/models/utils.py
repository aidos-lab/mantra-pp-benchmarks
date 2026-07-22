from torch import nn 
from torch_geometric.utils import scatter

READOUTS = ('mean', 'sum')


def pool(x, batch, reduce='mean'):
    """Aggregate node features into one vector per graph.

    `reduce='sum'` keeps the readout extensive, which matters for the
    Calabi-Yau targets: they scale with the size of the complex (e.g.
    h11 = f0 - 5 counts triangulation points), and a mean readout
    divides out exactly that quantity.
    """
    assert reduce in READOUTS, f"unknown readout {reduce!r}"
    return scatter(x, batch, dim=-2, reduce=reduce)


def mean_pool(x, batch):
    return scatter(x, batch, dim=-2, reduce='mean')

class SumReadout(nn.Module):
    def __init__(self, in_features, out_features):
        super(SumReadout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.projection_needed = in_features != out_features
        if self.projection_needed:
            self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, signal_belongings):
        if self.projection_needed:
            x = self.linear(x)
        x = scatter(x, signal_belongings, dim=-2, reduce='sum')
        return x