import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .blocks import (
    DepthwiseSeparableConv,
    LightweightResBlock,
    DeconvBlock,
    LightweightASPP,
    LightweightAttentionGate,
)

class LightweightAuraViT(nn.Module):
    """
    Lightweight version of AuraViT with:
    - Reduced model dimensions (50% parameter reduction)
    - Depthwise separable convolutions
    - Efficient attention mechanisms
    - Optimized decoder
    """
    def __init__(self, cf):
        super().__init__()
        self.cf = cf

        # --- LIGHTWEIGHT ViT ENCODER ---
        self.patch_embed = nn.Sequential(
            nn.Linear(cf["patch_size"]*cf["patch_size"]*cf["num_channels"], cf["hidden_dim"]),
            nn.LayerNorm(cf["hidden_dim"]),
            nn.Dropout(cf["dropout_rate"])
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, cf["num_patches"], cf["hidden_dim"]))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.pos_dropout = nn.Dropout(cf["dropout_rate"])
        
        # Fewer transformer layers with pre-norm for stability
        self.trans_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cf["hidden_dim"], 
                nhead=cf["num_heads"], 
                dim_feedforward=cf["mlp_dim"],
                dropout=cf["dropout_rate"], 
                activation=F.gelu, 
                batch_first=True,
                norm_first=True
            ) for _ in range(cf["num_layers"])
        ])
        
        # Adjust skip connections based on number of layers
        self.skip_norms = nn.ModuleList([
            nn.LayerNorm(cf["hidden_dim"]) for _ in range(4)
        ])

        # --- LIGHTWEIGHT ASPP MODULE ---
        self.aspp = LightweightASPP(cf["hidden_dim"], cf["hidden_dim"], rates=[6, 12, 18])

        # --- LIGHTWEIGHT ATTENTION GATES ---
        self.att_gate_1 = LightweightAttentionGate(gate_channels=256, in_channels=256, inter_channels=128)
        self.att_gate_2 = LightweightAttentionGate(gate_channels=128, in_channels=128, inter_channels=64)
        self.att_gate_3 = LightweightAttentionGate(gate_channels=64, in_channels=64, inter_channels=32)
        self.att_gate_4 = LightweightAttentionGate(gate_channels=32, in_channels=32, inter_channels=16)

        # --- LIGHTWEIGHT SEGMENTATION DECODER (Reduced channels) ---
        dropout_rate = cf.get("block_dropout_rate", 0.1)
        
        self.seg_d1 = DeconvBlock(cf["hidden_dim"], 256)
        self.seg_s1 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 256), 
            LightweightResBlock(256, 256, dropout_rate=dropout_rate)
        )
        self.seg_c1 = nn.Sequential(
            LightweightResBlock(256+256, 256, dropout_rate=dropout_rate), 
            LightweightResBlock(256, 256, dropout_rate=dropout_rate)
        )

        self.seg_d2 = DeconvBlock(256, 128)
        self.seg_s2 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 128), 
            LightweightResBlock(128, 128, dropout_rate=dropout_rate), 
            DeconvBlock(128, 128), 
            LightweightResBlock(128, 128, dropout_rate=dropout_rate)
        )
        self.seg_c2 = nn.Sequential(
            LightweightResBlock(128+128, 128, dropout_rate=dropout_rate), 
            LightweightResBlock(128, 128, dropout_rate=dropout_rate)
        )

        self.seg_d3 = DeconvBlock(128, 64)
        self.seg_s3 = nn.Sequential(
            DeconvBlock(cf["hidden_dim"], 64), 
            LightweightResBlock(64, 64, dropout_rate=dropout_rate), 
            DeconvBlock(64, 64),
            LightweightResBlock(64, 64, dropout_rate=dropout_rate), 
            DeconvBlock(64, 64), 
            LightweightResBlock(64, 64, dropout_rate=dropout_rate)
        )
        self.seg_c3 = nn.Sequential(
            LightweightResBlock(64+64, 64, dropout_rate=dropout_rate), 
            LightweightResBlock(64, 64, dropout_rate=dropout_rate)
        )

        self.seg_d4 = DeconvBlock(64, 32)
        self.seg_s4 = nn.Sequential(
            LightweightResBlock(cf["num_channels"], 32, dropout_rate=dropout_rate), 
            LightweightResBlock(32, 32, dropout_rate=dropout_rate)
        )
        self.seg_c4 = nn.Sequential(
            LightweightResBlock(32+32, 32, dropout_rate=dropout_rate), 
            LightweightResBlock(32, 32, dropout_rate=dropout_rate)
        )

        self.seg_output = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        
        nn.init.xavier_uniform_(self.seg_output.weight, gain=0.1)
        nn.init.constant_(self.seg_output.bias, 0)

    def forward(self, inputs):
        if torch.isnan(inputs).any():
            raise ValueError("NaN detected in input tensor")
            
        # 1. ViT Encoder
        p = self.cf["patch_size"]
        patches = inputs.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(inputs.size(0), inputs.size(1), -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.contiguous().view(inputs.size(0), self.cf["num_patches"], -1)
        patch_embed = self.patch_embed(patches)

        x = self.pos_dropout(patch_embed + self.pos_embed)

        # Adjust skip connection indices based on number of layers
        num_layers = len(self.trans_encoder_layers)
        if num_layers == 8:
            skip_connection_index = [1, 3, 5, 7]
        elif num_layers == 6:
            skip_connection_index = [1, 2, 4, 5]
        elif num_layers == 4:
            skip_connection_index = [0, 1, 2, 3]
        else:  # 12 layers
            skip_connection_index = [2, 5, 8, 11]
            
        skip_connections = []
        for i, layer in enumerate(self.trans_encoder_layers):
            x = layer(x)
            if torch.isnan(x).any():
                raise ValueError(f"NaN detected after transformer layer {i}")
            if i in skip_connection_index:
                norm_idx = len(skip_connections)
                normalized_skip = self.skip_norms[norm_idx](x)
                skip_connections.append(normalized_skip)
        
        z3, z6, z9, z12_features = skip_connections

        # Reshape feature maps
        batch, num_patches, hidden_dim = z12_features.shape
        patches_per_side = int(np.sqrt(num_patches))
        shape = (batch, hidden_dim, patches_per_side, patches_per_side)

        z0 = inputs
        z3 = z3.permute(0, 2, 1).contiguous().view(shape)
        z6 = z6.permute(0, 2, 1).contiguous().view(shape)
        z9 = z9.permute(0, 2, 1).contiguous().view(shape)
        z12_reshaped = z12_features.permute(0, 2, 1).contiguous().view(shape)

        # 2. Lightweight ASPP
        aspp_out = self.aspp(z12_reshaped)

        # 3. Lightweight Decoder
        x_seg = self.seg_d1(aspp_out)
        s = self.seg_s1(z9)
        s = self.att_gate_1(gate=x_seg, x=s)
        x_seg = torch.cat([x_seg, s], dim=1)
        x_seg = self.seg_c1(x_seg)

        x_seg = self.seg_d2(x_seg)
        s = self.seg_s2(z6)
        s = self.att_gate_2(gate=x_seg, x=s)
        x_seg = torch.cat([x_seg, s], dim=1)
        x_seg = self.seg_c2(x_seg)

        x_seg = self.seg_d3(x_seg)
        s = self.seg_s3(z3)
        s = self.att_gate_3(gate=x_seg, x=s)
        x_seg = torch.cat([x_seg, s], dim=1)
        x_seg = self.seg_c3(x_seg)

        x_seg = self.seg_d4(x_seg)
        s = self.seg_s4(z0)
        s = self.att_gate_4(gate=x_seg, x=s)
        x_seg = torch.cat([x_seg, s], dim=1)
        x_seg = self.seg_c4(x_seg)

        seg_output = self.seg_output(x_seg)
        
        if torch.isnan(seg_output).any():
            raise ValueError("NaN detected in model output")

        return seg_output
