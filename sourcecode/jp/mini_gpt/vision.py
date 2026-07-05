"""最小 Vision Encoder: 画像をパッチ単位のベクトルに変換"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """画像を patch_size x patch_size の断片に分割し、各断片をベクトルに投影"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        # 大きな stride を持つ 1 つの conv により「パッチ分割 + ベクトルへの投影」を同時に実行
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size,
        )
        self.position_embedding = nn.Embedding(self.num_patches, embed_dim)

    def forward(self, images):
        # images: (batch, channels, height, width)
        x = self.projection(images)                # (batch, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)              # (batch, num_patches, embed_dim)

        positions = torch.arange(self.num_patches, device=images.device).unsqueeze(0)
        x = x + self.position_embedding(positions)

        return x  # (batch, num_patches, embed_dim) - 画像が「トークンシーケンス」に変換される


class MiniVisionEncoder(nn.Module):
    """パッチ埋め込み + 数個の（テキストと同一の）Transformer ブロック"""

    def __init__(self, image_size=224, patch_size=16, embed_dim=256,
                 num_heads=8, num_layers=4):
        super().__init__()
        from mini_gpt.transformer_block import TransformerBlock

        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Vision Encoder は未来を隠す必要がないため（画像には順序がない）、
        # causal mask なしで全体を見渡せる Attention が理想的だが、
        # ここでは mini_gpt の既存ブロックをそのまま再利用するため、十分な大きさの max_seq_len のみを指定
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, max_seq_len=num_patches)
            for _ in range(num_layers)
        ])

    def forward(self, images):
        x = self.patch_embed(images)
        for block in self.blocks:
            x = block(x)
        return x  # (batch, num_patches, embed_dim)
class VisionToTextProjection(nn.Module):
    """Vision Encoder の出力を LLM の埋め込み次元に変換"""

    def __init__(self, vision_dim, text_embed_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or vision_dim * 2
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_embed_dim),
        )

    def forward(self, vision_features):
        return self.proj(vision_features)  # (batch, num_patches, text_embed_dim)
