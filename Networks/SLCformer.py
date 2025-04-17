import torch
from torch import nn, einsum
from einops import rearrange
from thop import profile

from .Utils.mask_utils import create_mask, get_relative_distances
from .Utils.patch_merging import PatchMerging

class WindowMSA(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size):
        super().__init__()
        inner_dim = head_dim * heads

        # Main attributes
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.shifted = shifted
        self.displacement = window_size // 2

        # Masks for shifted windows
        if self.shifted:
            self.upper_lower_mask = nn.Parameter(
                create_mask(window_size=window_size, displacement=self.displacement, upper_lower=True, left_right=False),
                requires_grad=False
            )
            self.left_right_mask = nn.Parameter(
                create_mask(window_size=window_size, displacement=self.displacement, upper_lower=False, left_right=True),
                requires_grad=False
            )

        # Linear projection for queries, keys, and values
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Relative position embedding
        self.relative_indices = get_relative_distances(window_size) + window_size - 1
        self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))

        # Output projection
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # Apply cyclic shift if enabled
        if self.shifted:
            x = torch.roll(x, shifts=(-self.displacement, -self.displacement), dims=(1, 2))

        # Input dimensions and QKV processing
        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
       
        # Window calculations
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        # Rearrange Q, K, V tensors
        q, k, v = map(
            lambda t: rearrange(
                t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                h=h, w_h=self.window_size, w_w=self.window_size
            ),
            qkv
        )

        # Scaled dot-product and relative position embedding
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]

        # Apply masks if shifted
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        # Attention calculation and output
        attn = dots.softmax(dim=-1)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(
            out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
            h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w
        )
        out = self.to_out(out)

        # Restore cyclic shift if enabled
        if self.shifted:
            out = torch.roll(out, shifts=(self.displacement, self.displacement), dims=(1, 2))

        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size):
        super().__init__()

        self.LN = nn.LayerNorm(dim)

        # Attention block (including pre-normalization)
        self.MSA = WindowMSA(
            dim=dim,
            heads=heads,
            head_dim=head_dim,
            shifted=shifted,
            window_size=window_size
        )

        self.MLP = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        #First Part
        x_normalized = self.LN(x)
        attention_out = self.MSA(x_normalized)
        x_residual = attention_out + x
        
        #Second Part
        x_normalized_mlp = self.LN(x_residual)
        mlp_out = self.MLP(x_normalized_mlp)
        output = mlp_out + x_residual
        
        return output


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size):
        super().__init__()

        # Ensure the number of layers is even for balanced regular and shifted blocks
        assert layers % 2 == 0, 'The stage layers must be divisible by 2 for both regular and shifted blocks.'

        # Initialize the patch partition layer
        self.patch_partition = PatchMerging(
            in_channels=in_channels, 
            out_channels=hidden_dimension, 
            downscaling_factor=downscaling_factor
        )
        
        # Prepare a list of SwinBlock layers
        self.layers = nn.ModuleList([])

        # Define each pair of regular and shifted SwinBlocks
        for _ in range(layers // 2):
            regular_block = SwinBlock(
                dim=hidden_dimension, 
                heads=num_heads, 
                head_dim=head_dim, 
                mlp_dim=hidden_dimension * 4,
                shifted=False, 
                window_size=window_size
            )
            shifted_block = SwinBlock(
                dim=hidden_dimension, 
                heads=num_heads, 
                head_dim=head_dim, 
                mlp_dim=hidden_dimension * 4,
                shifted=True, 
                window_size=window_size
            )
            
            # Append the pair of blocks to the layers list
            self.layers.append(nn.ModuleList([regular_block, shifted_block]))

    def forward(self, x):
        # First, partition the input into patches
        x = self.patch_partition(x)
        
        # Pass through each pair of regular and shifted SwinBlocks
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)  # Apply regular block
            x = shifted_block(x)  # Apply shifted block
        
        # Permute the output to match expected dimensions (N, C, H, W)
        return x.permute(0, 3, 1, 2)

        
class SLCFormer(nn.Module):
    def __init__(self, *, base_dim=32, num_layers=(2, 2, 6, 2), num_heads=(1, 2, 4, 8), input_channels=3, num_classes=2, 
                 head_dim=32, window_size=7, downscale_factors=(4, 2, 2, 2)):
                                                                
        super().__init__()
        
        # Initial stage for feature extraction with customizable parameters
        self.stage1 = StageModule(
            in_channels=input_channels,
            hidden_dimension=base_dim,
            layers=num_layers[0],
            downscaling_factor=downscale_factors[0],
            num_heads=num_heads[0],
            head_dim=head_dim,
            window_size=window_size
        )
        
        # Second stage with increased feature depth
        self.stage2 = StageModule(
            in_channels=base_dim,
            hidden_dimension=base_dim * 2,
            layers=num_layers[1],
            downscaling_factor=downscale_factors[1],
            num_heads=num_heads[1],
            head_dim=head_dim,
            window_size=window_size
        )
        
        # Third stage for deeper feature extraction
        self.stage3 = StageModule(
            in_channels=base_dim * 2,
            hidden_dimension=base_dim * 4,
            layers=num_layers[2],
            downscaling_factor=downscale_factors[2],
            num_heads=num_heads[2],
            head_dim=head_dim,
            window_size=window_size
        )
        
        # Final stage with the highest feature depth
        self.stage4 = StageModule(
            in_channels=base_dim * 4,
            hidden_dimension=base_dim * 8,
            layers=num_layers[3],
            downscaling_factor=downscale_factors[3],
            num_heads=num_heads[3],
            head_dim=head_dim,
            window_size=window_size
        )
        
        # Classification head: normalizes and maps features to the output classes
        self.finalStage = nn.Sequential(
            nn.LayerNorm(base_dim * 8),
            nn.Linear(base_dim * 8, num_classes)
        )

    def forward(self, input_image):
        # Pass through each stage
        features = self.stage1(input_image)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.stage4(features)
        
        # Global average pooling to reduce spatial dimensions
        features = features.mean(dim=[2, 3])  # Mean across height and width
        # Pass through the classification head for final output
        return self.finalStage(features)
    
    

