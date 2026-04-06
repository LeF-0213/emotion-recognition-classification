import torch
import torch.nn as nn
import timm
from transformers import BertModel

import config as C

class TextEncoder(nn.Module):
    def __init__(self, model_name: str, freeze_layers: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.output_dim = self.bert.config.hidden_size
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

class ImageEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, freeze_ratio: float):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.output_dim = self.backbone.num_features
        params = list(self.backbone.parameters())
        n_freeze = int(len(params) * freeze_ratio)
        for i, p in enumerate(params):
            if i < n_freeze:
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)

class ConcatFusionModel(nn.Module):
    def __init__(self, num_classes: int, dropout: float):
        super().__init__()
        self.text_enc = TextEncoder(C.TEXT_MODEL_NAME, C.TEXT_FREEZE_LAYERS)
        self.image_enc = ImageEncoder(C.IMAGE_BACKBONE, True, C.IMAGE_FREEZE_RATIO)
        fused_dim = self.text_enc.output_dim + self.image_enc.output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    def forward(self, input_ids, attention_mask, image):
        t = self.text_enc(input_ids, attention_mask)
        v = self.image_enc(image)
        z = self.fusion(torch.cat([t, v], dim=-1))
        return self.classifier(z)

class CrossModalFusionModel(nn.Module):
    def __init__(self, num_classes: int, dropout: float, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.text_enc = TextEncoder(C.TEXT_MODEL_NAME, C.TEXT_FREEZE_LAYERS)
        self.image_enc = ImageEncoder(C.IMAGE_BACKBONE, True, C.IMAGE_FREEZE_RATIO)
        self.proj_t = nn.Linear(self.text_enc.output_dim, hidden_dim)
        self.proj_v = nn.Linear(self.image_enc.output_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    def forward(self, input_ids, attention_mask, image):
        t = self.text_enc(input_ids, attention_mask)
        v = self.image_enc(image)
        x = torch.stack([self.proj_t(t), self.proj_v(v)], dim=1)
        y, _ = self.mha(x, x, x)
        x = self.norm(x + y)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)

def build_model(name: str):
    name = name.lower()
    if name in ("concat", "concatfusion"):
        return ConcatFusionModel(C.NUM_CLASSES, C.DROPOUT)
    if name in ("cross", "crossmodal", "attention"):
        return CrossModalFusionModel(C.NUM_CLASSES, C.DROPOUT)
    raise ValueError(name)
