from .integral_operators import OperatorBlock_2D
from .hpf import Sobel
import torch.nn as nn
import torch
from einops import rearrange
from .galerkin import simple_attn
import torch.nn.functional as F
from torchinfo import summary
from ptflops import get_model_complexity_info
import math
import numpy as np
class Unet2D_FNO(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_channels: list = [64, 128, 256, 512, 1024],
        target_size: list = [256, 256],
        trunc_mode_stages: list = None,
        use_sobel_stages: list = None,
        patch_based_stages: list = None,
        patch_size_stages: list= None,
        factorize_mode_stages: list = None,
        use_attn_stages: list = None,
        type_grid: str = 'linear' # [None, 'linear']
    ):
        super(Unet2D_FNO,self).__init__()
        self.in_channels = in_channels
        self.target_size = target_size
        self.height, self.width = target_size
        self.trunc_mode_stages = trunc_mode_stages
        self.use_sobel_stages = use_sobel_stages
        self.patch_based_stages = patch_based_stages
        self.patch_size_stages = patch_size_stages
        self.factorize_mode_stages = factorize_mode_stages
        self.use_attn_stages = use_attn_stages
        self.type_grid = type_grid
        # Initialize layers
        self.use_sobel = any(use_sobel_stages)  # Enable Sobel if any stage uses it
        self.sobel = Sobel() if self.use_sobel else None
        if type_grid == None:
            self.lift00 = nn.Linear(in_channels, num_channels[0]//2)
        else:
            self.lift00 = nn.Linear(in_channels + 2, num_channels[0]//2)
        self.lift01 = nn.Linear(num_channels[0]//2, num_channels[0])
        self.down = nn.ModuleList([])
        for i in range(1, len(num_channels)):
            dim1, dim2 = self.height // 2**(i), self.height // 2**(i)
            if patch_based_stages[i-1]:
                modes1_patch, modes2_patch = self.patch_size_stages[i-1], patch_size_stages[i-1] // 2+1
            else:
                modes1_patch, modes2_patch = None, None
            if trunc_mode_stages[i-1] == "shared_sliding":
                modes1, modes2 = dim1 // 8 , dim2 // 16 # since weight are shared anyway, use smaller modes
            else:
                modes1, modes2 = dim1 // 2, dim2 // 2
            # Use attention only for the last layer
            self.down.append(OperatorBlock_2D(
                in_codim=num_channels[i - 1],
                out_codim=num_channels[i],
                dim1=dim1, 
                dim2=dim2,
                modes1=modes1, 
                modes2=modes2,
                modes1_patch=modes1_patch,
                modes2_patch=modes2_patch,
                trunc_mode=trunc_mode_stages[i-1],
                use_attn=use_attn_stages[i-1],
                use_sobel=use_sobel_stages[i-1],
                patch_based=patch_based_stages[i-1],
                patch_size=patch_size_stages[i-1],
                factorize_mode=factorize_mode_stages[i-1]
            ))
 
        # self.mid = OperatorBlock_2D(
        #     num_channels[-1],
        #     num_channels[-1],
        #     dim1, dim2,
        #     modes1, modes2
        # )
    def forward(self, x): # x: [B*M, C, W, H]
        BM, C, W, H = x.shape
        grid = self.get_grid(x.shape, x.device) if self.type_grid is not None else None
        h = []
        # Sobel filtering
        x_sobel = self.sobel(x) if self.use_sobel else None
        # Concat grid
        if self.type_grid is not None:
            x = torch.cat((x, grid), dim=1)
        # Initial lifting
        x = x.permute(0, 2, 3, 1).contiguous() # x: [B*M, W, H, C]
        x = self.lift00(x) # x: [B*M, W, H, num_channels[0]//2]
        x = F.gelu(x)

        x = self.lift01(x) # x: [B*M, W, H, num_channels[0]//2]
        x = F.gelu(x)
        x = x.permute(0, 3, 1, 2).contiguous() # x: [B*M, num_channels[0], W, H]
        # down
        for i, block in enumerate(self.down):
            x = block(x, x_sobel) if self.use_sobel else block(x)
            h.append(x)
        # mid
        # x = self.mid(x)
        # h.append(x)
        return x, h
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[-2], shape[-1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

    def get_model_info(self, input_size=None, verbose=0):
        """
        Compute parameters, FLOPs, and memory for the Unet2D_FNO model using torchinfo and ptflops.
        
        Args:
            input_size (tuple, optional): Input shape (e.g., (1, in_channels, H, W)).
                                         Defaults to (1, self.in_channels, target_size[0], target_size[1]).
        
        Returns:
            dict: Dictionary containing:
                - total_params: Total number of parameters.
                - learnable_params: Number of learnable parameters.
                - non_learnable_params: Number of non-learnable parameters.
                - flops_gflops: FLOPs in gigaflops (GFLOPs).
                - memory_mb: Estimated memory in MB for saving state_dict.
                - summary: Detailed layer-wise summary from torchinfo.
        """
        if input_size is None:
            input_size = (1, self.in_channels, self.target_size[0], self.target_size[1])

        # Get parameter counts and summary using torchinfo
        try:
            model_summary = summary(self, input_size=input_size,col_names=["input_size","output_size","num_params"],verbose=verbose)
            total_params = model_summary.total_params
            learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            non_learnable_params = total_params - learnable_params
            summary_str = str(model_summary)
        except Exception as e:
            print(f"Error in torchinfo: {e}")
            total_params = sum(p.numel() for p in self.parameters())
            learnable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            non_learnable_params = total_params - learnable_params
            summary_str = "Summary unavailable due to error."

        # Get FLOPs using ptflops
        try:
            flops, _ = get_model_complexity_info(
                self,
                input_size[1:],  # Remove batch dimension
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False
            )
            flops_gflops = flops / 1e9  # Convert to GFLOPs
        except Exception as e:
            print(f"Error in ptflops: {e}")
            flops_gflops = 0.0

        # Estimate memory (4 bytes per parameter, convert to MB)
        memory_mb = total_params * 4 / 1e6  # 4 bytes per parameter, MB = 1e6 bytes

        return {
            "total_params": total_params,
            "learnable_params": learnable_params,
            "non_learnable_params": non_learnable_params,
            "flops_gflops": flops_gflops,
            "memory_mb": memory_mb,
            "summary": summary_str
        }