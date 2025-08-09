from pathlib import Path

import torch
import torch.nn as nn
import math

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = n_track * 2 * 2
        output_dim = n_waypoints * 2

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = track_left.shape[0]
        track_left_flat = track_left.reshape(batch_size, -1)
        track_right_flat = track_right.reshape(batch_size, -1)
        x = torch.cat([track_left_flat, track_right_flat], dim=-1)
        output = self.mlp(x)
        waypoints = output.reshape(batch_size, self.n_waypoints, 2)
        return waypoints


class TransformerPlanner(nn.Module):
    """
    Transformer-based planner using Perceiver-like learned queries, but with
    an encoder for the lane boundary points for stronger context.
    """

    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        input_dim: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model
        self.max_lane_points = 2 * n_track

        # Learned waypoint queries (latent)
        self.waypoint_queries = nn.Embedding(n_waypoints, d_model)

        # Input projection for per-point features (lane points)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encodings (learnable)
        self.lane_pos_encoding = nn.Parameter(torch.randn(self.max_lane_points, d_model) * 0.02)
        self.waypoint_pos_encoding = nn.Parameter(torch.randn(n_waypoints, d_model) * 0.02)

        # Transformer encoder to encode lane features (memory)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Transformer decoder layers for cross-attention from queries -> memory
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Head: small MLP to coordinates with residual from queries
        self.pre_out_ln = nn.LayerNorm(d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        # small MLP to transform queries before residual addition
        self.query_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.waypoint_queries.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        # lane_boundaries: (batch, 2*n_track, 2)
        lane_boundaries = torch.cat([track_left, track_right], dim=1)
        batch_size, num_points, _ = lane_boundaries.shape

        if num_points > self.max_lane_points:
            lane_boundaries = lane_boundaries[:, :self.max_lane_points]
            num_points = self.max_lane_points

        # Project lane points -> d_model
        lane_features = self.input_projection(lane_boundaries)  # (b, points, d_model)
        # Add positional enc
        lane_features = lane_features + self.lane_pos_encoding[:num_points].unsqueeze(0)
        lane_features = self.dropout(lane_features)

        # Encode lane features with a short transformer encoder
        # (batch, points, d_model)
        encoded_memory = self.transformer_encoder(lane_features)

        # Queries
        waypoint_idx = torch.arange(self.n_waypoints, device=lane_boundaries.device)
        queries = self.waypoint_queries(waypoint_idx).unsqueeze(0).expand(batch_size, -1, -1)  # (b, n_waypoints, d_model)
        queries = queries + self.waypoint_pos_encoding.unsqueeze(0)
        queries = self.dropout(queries)

        # Decoder: queries attend to encoded memory (cross-attention)
        # No padding mask here since we assume fixed-length lane points
        attended = self.transformer_decoder(tgt=queries, memory=encoded_memory)  # (b, n_waypoints, d_model)

        # Residual + small MLP on original queries to stabilize outputs
        queries_res = self.query_mlp(queries)
        combined = attended + queries_res  # residual

        # Normalize and project to coordinates
        combined = self.pre_out_ln(combined)
        waypoints = self.output_projection(combined)  # (b, n_waypoints, 2)

        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> torch.nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    model_size_mb = calculate_model_size_mb(m)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    return m


def save_model(model: torch.nn.Module) -> str:
    model_name = None
    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n
    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")
    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
