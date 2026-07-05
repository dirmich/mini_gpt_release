"""최소 Vision Encoder: 이미지를 패치 단위 벡터로 변환"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """이미지를 patch_size x patch_size 조각으로 나누고 각 조각을 벡터로 투영"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        # 큰 stride의 conv 하나로 "패치 나누기 + 벡터로 투영"을 동시에 수행
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

        return x  # (batch, num_patches, embed_dim) - 이미지가 "토큰 시퀀스"로 변환됨


class MiniVisionEncoder(nn.Module):
    """패치 임베딩 + 몇 개의 (텍스트와 동일한) 트랜스포머 블록"""

    def __init__(self, image_size=224, patch_size=16, embed_dim=256,
                 num_heads=8, num_layers=4):
        super().__init__()
        from mini_gpt.transformer_block import TransformerBlock

        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 비전 인코더는 미래를 가릴 필요가 없으므로(이미지는 순서가 없음),
        # causal mask 없이 전체를 다 볼 수 있는 어텐션이 이상적이지만
        # 여기서는 mini_gpt의 기존 블록을 그대로 재사용하기 위해 충분히 큰 max_seq_len만 지정
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
    """비전 인코더 출력을 LLM의 임베딩 차원으로 변환"""

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
