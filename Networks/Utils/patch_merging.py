from torch import nn

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        
        # Class parameters
        self.downscaling_factor = downscaling_factor
        
        # Unfold operation to extract patches from the image
        self.patch_merge = nn.Unfold(
            kernel_size=downscaling_factor, 
            stride=downscaling_factor, 
            padding=0
        )
        
        # Linear layer to transform patch dimensions
        self.linear = nn.Linear(
            in_channels * (downscaling_factor ** 2), 
            out_channels
        )

    def forward(self, x):
        # Input dimensions: [batch, channels, height, width]
        b, c, h, w = x.shape

        # New dimensions after downscaling
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        
        # Apply nn.Unfold to split the image into patches
        # Output shape after patch_merge: [batch, num_patches * patch_size, num_windows]
        x = self.patch_merge(x)  # Shape: [b, c * patch_size, num_patches]

        # Rearrange the output to [batch, num_patches_h, num_patches_w, c * patch_size]
        x = x.view(b, -1, new_h, new_w).permute(0, 2, 3, 1)

        # Apply the linear layer to transform patch dimensions
        x = self.linear(x)
        
        # Final output shape: [batch, num_patches_h, num_patches_w, out_channels]
        return x

