from torch import nn

from src.models.set_transformer import SetTransformer


class MaximalFaceSetTransformer(nn.Module):
    """Set Transformer over learned embeddings of maximal triangular faces."""

    def __init__(
        self,
        in_channels,
        num_outputs,
        out_channels,
        num_inds,
        hidden_channels,
        num_heads,
        ln=False,
    ):
        super().__init__()
        self.vertex_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.face_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.face_transformer = SetTransformer(
            in_channels=hidden_channels,
            num_outputs=num_outputs,
            out_channels=out_channels,
            num_inds=num_inds,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            ln=ln,
        )

    def forward(self, batch):
        vertex_features = self.vertex_encoder(batch["x_0"])
        face_features = vertex_features[batch["top_simplex_index"]].mean(dim=0)
        face_features = self.face_encoder(face_features)
        return self.face_transformer(
            {"x": face_features, "batch": batch["batch_2"]}
        )
