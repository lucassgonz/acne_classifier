import torch
import torch.nn as nn
from torchvision import models


class OptimizedMultiRegionResNet18(nn.Module):
    def __init__(self, num_classes=4, num_regions=5, pretrained_backbone_path=None):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        if pretrained_backbone_path is not None:
            state = torch.load(pretrained_backbone_path, map_location="cpu")
            self.resnet.load_state_dict(state, strict=False)
        self.region_embed = nn.Embedding(num_regions, 64)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, region_idx):
        img_features = self.resnet(x)
        region_features = self.region_embed(region_idx.long()).squeeze(1)
        combined = torch.cat([img_features, region_features], dim=1)
        return self.classifier(combined)
