import torch
import torch.nn as nn
import torchvision.models as models


class ConvAdapter(nn.Module):
    """
    Small task-specific convolutional refinement block.
    Lightweight for 4GB GPU.
    """
    def __init__(self, in_channels=1280):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MultiTaskModel(nn.Module):
    def __init__(self, freeze_backbone=False):
        super().__init__()

        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        self.features = backbone.features  # shared backbone

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Task-specific spatial adapters
        self.gender_adapter = ConvAdapter(1280)
        self.smile_adapter = ConvAdapter(1280)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Task-specific classifiers
        self.gender_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

        self.smile_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        shared = self.features(x)  # (B, 1280, H, W)

        # Task-specific spatial refinement
        gender_feat = self.gender_adapter(shared)
        smile_feat = self.smile_adapter(shared)

        # Pool separately
        gender_feat = self.pool(gender_feat)
        smile_feat = self.pool(smile_feat)

        gender_out = self.gender_head(gender_feat)
        smile_out = self.smile_head(smile_feat)

        return torch.cat([gender_out, smile_out], dim=1)