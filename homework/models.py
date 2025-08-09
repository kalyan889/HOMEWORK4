from pathlib import Path

import torch
import torch.nn as nn

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
        """
        MLP-based planner that predicts waypoints from track boundaries.
        
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): dimension of hidden layers
            num_layers (int): number of hidden layers
            dropout (float): dropout probability
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        # Input dimension: left track (n_track * 2) + right track (n_track * 2)
        input_dim = n_track * 2 * 2
        
        # Output dimension: waypoints (n_waypoints * 2)
        output_dim = n_waypoints * 2
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]
        
        # Flatten and concatenate track boundaries
        # track_left: (b, n_track, 2) -> (b, n_track * 2)
        # track_right: (b, n_track, 2) -> (b, n_track * 2)
        track_left_flat = track_left.reshape(batch_size, -1)
        track_right_flat = track_right.reshape(batch_size, -1)
        
        # Concatenate left and right tracks
        # Combined: (b, n_track * 2 * 2)
        x = torch.cat([track_left_flat, track_right_flat], dim=-1)
        
        # Pass through MLP
        # Output: (b, n_waypoints * 2)
        output = self.mlp(x)
        
        # Reshape to (b, n_waypoints, 2)
        waypoints = output.reshape(batch_size, self.n_waypoints, 2)
        
        return waypoints


class TransformerPlanner(nn.Module):
    """
    Transformer-based planner using Perceiver-like architecture.
    Uses learned waypoint query embeddings to attend over track boundary features.
    """
    
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        input_dim: int = 2,  # x, y coordinates
        activation: str = "relu"
    ):
        """
        Args:
            n_track: Number of track boundary points per side (left/right)
            n_waypoints: Number of waypoints to predict
            d_model: Dimension of the model embeddings
            nhead: Number of attention heads
            num_decoder_layers: Number of transformer decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            input_dim: Dimension of input coordinates (x, y)
            activation: Activation function for transformer layers
        """
        super().__init__()
        
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model
        self.max_lane_points = 2 * n_track  # left + right track boundaries
        
        # Learned waypoint query embeddings (latent array in Perceiver terms)
        self.waypoint_queries = nn.Embedding(n_waypoints, d_model)
        
        # Input projection for track boundary points (byte array projection)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding for track boundary points
        self.lane_pos_encoding = nn.Parameter(
            torch.randn(self.max_lane_points, d_model) * 0.02
        )
        
        # Positional encoding for waypoint queries
        self.waypoint_pos_encoding = nn.Parameter(
            torch.randn(n_waypoints, d_model) * 0.02
        )
        
        # Transformer decoder layers for cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output projection to waypoint coordinates
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 2)  # x, y coordinates
        )
        
        # Dropout for embeddings
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize waypoint queries with small random values
        nn.init.normal_(self.waypoint_queries.weight, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.
        
        Args:
            track_left: Left track boundary points, shape (batch_size, n_track, 2)
            track_right: Right track boundary points, shape (batch_size, n_track, 2)
        
        Returns:
            waypoints: Predicted waypoints, shape (batch_size, n_waypoints, 2)
        """
        # Concatenate left and right track boundaries
        lane_boundaries = torch.cat([track_left, track_right], dim=1)  # (batch, 2*n_track, 2)
        batch_size, num_points, _ = lane_boundaries.shape
        
        # Ensure we don't exceed max_lane_points
        if num_points > self.max_lane_points:
            lane_boundaries = lane_boundaries[:, :self.max_lane_points]
            num_points = self.max_lane_points
        
        # Project track boundary points to model dimension (byte array -> embeddings)
        lane_features = self.input_projection(lane_boundaries)  # (batch, points, d_model)
        
        # Add positional encoding to lane features
        lane_features = lane_features + self.lane_pos_encoding[:num_points].unsqueeze(0)
        lane_features = self.dropout(lane_features)
        
        # Get waypoint query embeddings (latent array)
        waypoint_indices = torch.arange(self.n_waypoints, device=lane_boundaries.device)
        waypoint_queries = self.waypoint_queries(waypoint_indices)  # (n_waypoints, d_model)
        waypoint_queries = waypoint_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_waypoints, d_model)
        
        # Add positional encoding to waypoint queries
        waypoint_queries = waypoint_queries + self.waypoint_pos_encoding.unsqueeze(0)
        waypoint_queries = self.dropout(waypoint_queries)
        
        # Create padding mask (no padding for track boundaries - all points are valid)
        memory_key_padding_mask = torch.zeros(
            batch_size, num_points, 
            device=lane_boundaries.device,
            dtype=torch.bool
        )
        
        # Apply transformer decoder layers (cross-attention)
        # tgt: waypoint queries, memory: lane features
        attended_waypoints = self.transformer_decoder(
            tgt=waypoint_queries,
            memory=lane_features,
            memory_key_padding_mask=memory_key_padding_mask
        )  # (batch, n_waypoints, d_model)
        
        # Project to waypoint coordinates
        waypoints = self.output_projection(attended_waypoints)  # (batch, n_waypoints, 2)
        
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
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        raise NotImplementedError


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
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

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
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
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024