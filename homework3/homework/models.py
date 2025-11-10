from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Convolutional layers
        # Input: (B, 3, 64, 64)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # After conv1 + pool: (B, 32, 32, 32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # After conv2 + pool: (B, 64, 16, 16)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # After conv3 + pool: (B, 128, 8, 8)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # After conv4 + pool: (B, 256, 4, 4)

        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        # After flattening: (B, 256 * 4 * 4) = (B, 4096)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Conv block 1: (B, 3, 64, 64) -> (B, 32, 32, 32)
        z = self.conv1(z)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.pool(z)

        # Conv block 2: (B, 32, 32, 32) -> (B, 64, 16, 16)
        z = self.conv2(z)
        z = self.bn2(z)
        z = self.relu(z)
        z = self.pool(z)

        # Conv block 3: (B, 64, 16, 16) -> (B, 128, 8, 8)
        z = self.conv3(z)
        z = self.bn3(z)
        z = self.relu(z)
        z = self.pool(z)

        # Conv block 4: (B, 128, 8, 8) -> (B, 256, 4, 4)
        z = self.conv4(z)
        z = self.bn4(z)
        z = self.relu(z)
        z = self.pool(z)

        # Flatten: (B, 256, 4, 4) -> (B, 4096)
        z = z.view(z.size(0), -1)

        # Fully connected layers
        z = self.fc1(z)
        z = self.relu(z)
        z = self.dropout(z)

        logits = self.fc2(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class DownConvBlock(nn.Module):
    """Down-sampling convolutional block with stride 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConvBlock(nn.Module):
    """Up-sampling convolutional block with transposed convolution"""
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # After concatenating with skip connection, we have out_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.upconv(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Detector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Initial convolution (no downsampling)
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Encoder (downsampling path)
        self.down1 = DownConvBlock(16, 32)    # h/2, w/2
        self.down2 = DownConvBlock(32, 64)    # h/4, w/4
        self.down3 = DownConvBlock(64, 128)   # h/8, w/8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Decoder (upsampling path) with skip connections
        self.up1 = UpConvBlock(128, 64, skip_channels=64)  # h/4, w/4
        self.up2 = UpConvBlock(64, 32, skip_channels=32)   # h/2, w/2
        self.up3 = UpConvBlock(32, 16, skip_channels=16)   # h, w

        # Segmentation head
        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)

        # Depth head with sigmoid to constrain output to [0, 1]
        self.depth_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Encoder path with skip connections
        x0 = self.input_conv(z)        # (b, 16, h, w)
        x1 = self.down1(x0)            # (b, 32, h/2, w/2)
        x2 = self.down2(x1)            # (b, 64, h/4, w/4)
        x3 = self.down3(x2)            # (b, 128, h/8, w/8)

        # Bottleneck
        x = self.bottleneck(x3)        # (b, 128, h/8, w/8)

        # Decoder path with skip connections
        x = self.up1(x, x2)            # (b, 64, h/4, w/4)
        x = self.up2(x, x1)            # (b, 32, h/2, w/2)
        x = self.up3(x, x0)            # (b, 16, h, w)

        # Task-specific heads
        logits = self.seg_head(x)      # (b, num_classes, h, w)
        depth = self.depth_head(x)     # (b, 1, h, w)
        depth = depth.squeeze(1)       # (b, h, w)

        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
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
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing Classifier...")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)
    print(f"  Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: ({batch_size}, 6)")

    print("\nTesting Detector...")
    # Test with the expected input size (96, 128)
    sample_batch = torch.rand(batch_size, 3, 96, 128).to(device)
    print(f"  Input shape: {sample_batch.shape}")

    model = load_model("detector", in_channels=3, num_classes=3).to(device)
    logits, depth = model(sample_batch)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, 3, 96, 128)")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Expected: ({batch_size}, 96, 128)")
    print(f"  Depth min/max: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"  Expected: [0.0, 1.0] (normalized)")

    # Test with arbitrary input size
    print("\nTesting with arbitrary input size...")
    sample_batch = torch.rand(batch_size, 3, 128, 256).to(device)
    print(f"  Input shape: {sample_batch.shape}")
    logits, depth = model(sample_batch)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Depth shape: {depth.shape}")

    # Test predict function
    print("\nTesting predict function...")
    pred, depth_pred = model.predict(sample_batch)
    print(f"  Prediction shape: {pred.shape}")
    print(f"  Prediction unique values: {pred.unique().tolist()}")
    print(f"  Depth prediction shape: {depth_pred.shape}")
    print(f"  Depth prediction min/max: [{depth_pred.min():.4f}, {depth_pred.max():.4f}]")


if __name__ == "__main__":
    debug_model()
