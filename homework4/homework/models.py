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
        hidden_dim: int = 512,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): dimension of hidden layers
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Input: concatenate left and right track boundaries
        # (n_track, 2) + (n_track, 2) = (n_track * 2, 2) -> flatten to (n_track * 4)
        input_dim = n_track * 4
        output_dim = n_waypoints * 2

        # Improved MLP architecture with batch normalization and dropout
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, output_dim),
        )

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

        # Concatenate left and right boundaries and flatten
        # (b, n_track, 2) + (b, n_track, 2) -> (b, n_track * 4)
        x = torch.cat([track_left, track_right], dim=1)  # (b, 2*n_track, 2)
        x = x.reshape(batch_size, -1)  # (b, n_track * 4)

        # Pass through MLP
        x = self.mlp(x)  # (b, n_waypoints * 2)

        # Reshape to waypoints
        waypoints = x.reshape(batch_size, self.n_waypoints, 2)  # (b, n_waypoints, 2)

        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        """
        Transformer-based planner using cross-attention.
        Uses learned query embeddings to attend over lane boundary features.

        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            d_model (int): dimension of transformer model
            nhead (int): number of attention heads
            num_layers (int): number of transformer decoder layers
            dim_feedforward (int): dimension of feedforward network
            dropout (float): dropout rate
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Learned query embeddings for waypoints (like Perceiver's latent array)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Encode input track boundaries to d_model dimensions
        # Input: 2D points (x, z) from left and right boundaries
        self.input_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding for track boundary points
        self.pos_encoder = nn.Parameter(torch.randn(1, n_track * 2, d_model))

        # Transformer decoder for cross-attention
        # Queries: waypoint embeddings
        # Memory (keys/values): encoded track boundaries
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Output head to predict waypoint coordinates
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),  # Output: (x, z) coordinates
        )

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

        # Concatenate left and right track boundaries
        # (b, n_track, 2) + (b, n_track, 2) -> (b, 2*n_track, 2)
        track_points = torch.cat([track_left, track_right], dim=1)

        # Encode track boundary points to d_model dimensions
        # (b, 2*n_track, 2) -> (b, 2*n_track, d_model)
        memory = self.input_encoder(track_points)

        # Add positional encoding
        memory = memory + self.pos_encoder

        # Get query embeddings for waypoints
        # (n_waypoints, d_model) -> (b, n_waypoints, d_model)
        query_indices = torch.arange(self.n_waypoints, device=track_left.device)
        queries = self.query_embed(query_indices).unsqueeze(0).expand(batch_size, -1, -1)

        # Apply transformer decoder (cross-attention)
        # Queries attend over memory (track boundaries)
        # (b, n_waypoints, d_model)
        decoder_output = self.transformer_decoder(tgt=queries, memory=memory)

        # Predict waypoint coordinates
        # (b, n_waypoints, d_model) -> (b, n_waypoints, 2)
        waypoints = self.output_head(decoder_output)

        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        num_channels: tuple = (32, 64, 128, 256),
    ):
        """
        CNN-based planner that predicts waypoints directly from images.

        Args:
            n_waypoints (int): number of waypoints to predict
            num_channels (tuple): number of channels in each CNN block
        """
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # CNN backbone - downsample image and extract features
        # Input: (B, 3, 96, 128)
        c1, c2, c3, c4 = num_channels

        self.backbone = nn.Sequential(
            # Block 1: (B, 3, 96, 128) -> (B, 32, 48, 64)
            nn.Conv2d(3, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),

            # Block 2: (B, 32, 48, 64) -> (B, 64, 24, 32)
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),

            # Block 3: (B, 64, 24, 32) -> (B, 128, 12, 16)
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),

            # Block 4: (B, 128, 12, 16) -> (B, 256, 6, 8)
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),

            # Block 5: (B, 256, 6, 8) -> (B, 256, 3, 4)
            nn.Conv2d(c4, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        # Global average pooling: (B, 256, 3, 4) -> (B, 256, 1, 1) -> (B, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Regression head to predict waypoints
        # (B, 256) -> (B, n_waypoints * 2)
        self.regression_head = nn.Sequential(
            nn.Linear(c4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = image.shape[0]

        # Normalize image
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Extract features with CNN backbone
        x = self.backbone(x)  # (B, 256, 3, 4)

        # Global pooling
        x = self.global_pool(x)  # (B, 256, 1, 1)
        x = x.view(batch_size, -1)  # (B, 256)

        # Predict waypoints
        x = self.regression_head(x)  # (B, n_waypoints * 2)

        # Reshape to waypoints
        waypoints = x.view(batch_size, self.n_waypoints, 2)  # (B, n_waypoints, 2)

        return waypoints


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
