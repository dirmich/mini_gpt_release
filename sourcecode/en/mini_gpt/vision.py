"""Minimal Vision Encoder: Converts images into patch-level vectors"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Divides the image into patch_size x patch_size pieces and projects each piece into a vector"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        # Performs "patch division + projection to vector" simultaneously using a single conv with a large stride
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

        return x  # (batch, num_patches, embed_dim) - The image is converted into a "token sequence"


class MiniVisionEncoder(nn.Module):
    """Patch embedding + several (identical to text) Transformer blocks"""

    def __init__(self, image_size=224, patch_size=16, embed_dim=256,
                 num_heads=8, num_layers=4):
        super().__init__()
        from mini_gpt.transformer_block import TransformerBlock

        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Since the vision encoder does not need to mask the future (images have no order),
        # attention that can see everything without a causal mask is ideal, but
        # here we only specify a sufficiently large max_seq_len to reuse mini_gpt's existing blocks as-is
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
    """Converts the vision encoder output to the LLM's embedding dimension"""

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
