import torch
import torch.nn as nn

class TorchTransformer(nn.Module):
    def __init__(self, input_dim=1024, embed_dim=512, num_heads=8, ff_dim=2048, num_classes=2, dropout=0.1):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B, N, D = x.shape  # [batch_size, num_patches, feature_dim]
        x = self.embedding(x)

        # Add [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # [B, N+1, embed_dim]

        x = self.transformer(x)
        cls_output = x[:, 0]  # use the [CLS] token output

        return self.classifier(cls_output)
