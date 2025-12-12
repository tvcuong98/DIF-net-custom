import os
import shutil  # CHANGE 1: Added for config file backup
import argparse
import numpy as np
import json  # CHANGE 1: Added for saving args
import subprocess  # CHANGE 1: Added for Git operations
import wandb  # CHANGE 1: Added for wandb integration
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
 
from dataset import Mixed_CBCT_dataset
from models.model import DIF_Net
from utils import convert_cuda, add_argument
from evaluate import eval_one_epoch
from tqdm import tqdm  # Added tqdm import
 
from tabulate import tabulate  # CHANGE 1: Added for model info table display
 
 
def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)
# CHANGE 1: Added function to retrieve Git branch and commit hash
def get_git_info():
    """Retrieve the current Git branch and short commit hash."""
    try:
        # Get the current branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True
        ).strip()
 
        # Get the short-hand commit hash (first 7 characters)
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True
        ).strip()
 
        return {"branch": branch, "commit": commit}
    except subprocess.CalledProcessError as e:
        return {"error": f"Failed to get Git info: {e}"}
    except FileNotFoundError:
        return {"error": "Git is not installed or not found in PATH"}
# CHANGE 1: Added function to display and save model information
def display_model_info(info, save_path=None):
    """
    Display model information in a formatted way and optionally save to a file.
   
    Args:
        info (dict): Dictionary from get_model_info containing model metrics.
        save_path (str, optional): Path to save the output as a text or JSON file.
    """
    # Extract key metrics for tabular display
    table_data = [
        ["Total Parameters", f"{info['total_params']:,}"],
        ["Learnable Parameters", f"{info['learnable_params']:,}"],
        ["Non-Learnable Parameters", f"{info['non_learnable_params']:,}"],
        ["FLOPs (GFLOPs)", f"{info['flops_gflops']:.2f}"],
        ["Memory (MB)", f"{info['memory_mb']:.2f}"]
    ]
   
    # Print table
    print("\n=== Model Information ===")
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
   
    # # Print dictionary using pprint for structured view
    # print("\n=== Detailed Metrics ===")
    # pprint({k: v for k, v in info.items() if k != "summary"}, indent=2)
   
    # # Print summary header (first few lines of torchinfo summary)
    # print("\n=== Model Summary (Preview) ===")
    # summary_lines = info["summary"].split("\n")
    # print("\n".join(summary_lines[:10]))  # Show first 10 lines
    # if len(summary_lines) > 10:
    #     print("... (full summary truncated, see saved file or increase preview limit)")
   
    # Save to file if save_path is provided
    if save_path:
        if save_path.endswith(".json"):
            with open(save_path, "w") as f:
                json.dump(info, f, indent=4)
            print(f"\nSaved model info as JSON to {save_path}")
        else:
            with open(save_path, "w") as f:
                f.write("=== Model Information ===\n")
                f.write(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
                f.write("\n\n=== Detailed Metrics ===\n")
                f.write(json.dumps({k: v for k, v in info.items() if k != "summary"}, indent=2))
                f.write("\n\n=== Model Summary ===\n")
                f.write(info["summary"])
            print(f"\nSaved model info as text to {save_path}")
 
def parse_none(value):
    """Convert a string input to a list, turning 'none' or 'null' elements to None."""
    return None if value.strip().lower() in ["none", "null"] else value
def str_to_bool(value):
    """Convert a string to a boolean."""
    value = value.strip().lower()
    if value in ["true", "1", "t", "y", "yes"]:
        return True
    elif value in ["false", "0", "f", "n", "no"]:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")
def str_to_int(value):
    """Convert a string to an integer, handling 'none' or 'null' as None."""
    if value.strip().lower() in ["none", "null"]:
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser = add_argument(parser)
    # Additional params
    parser.add_argument('-trunc','--trunc_mode_stages', type=parse_none,nargs='+', default=None, help="Comma-separated list of trunc modes (e.g., 'LL-LH,LH-HH,shared_sliding') or 'None'")
    parser.add_argument('-sobel','--use_sobel_stages', type=str_to_bool, nargs='+', default=None, help="List of booleans (e.g., 'True False True') or 'None'")
    parser.add_argument('-patch','--patch_based_stages', type=str_to_bool, nargs='+', default=None, help="List of booleans (e.g., 'True False True') or 'None'")
    parser.add_argument('-psize','--patch_size_stages', type=str_to_int, default=None,nargs='+', help="Patch size as an integer, or None")
    parser.add_argument('-fac','--factorize_mode_stages', type=parse_none,nargs='+', default=None, help="Comma-separated list of factorize modes (e.g., 'mode1,mode2,mode3') or 'None'")
    parser.add_argument('-attn','--use_attn_stages', type=str_to_bool, nargs='+', default=None, help="List of booleans (e.g., 'True False True') or 'None'")
    parser.add_argument('-grid','--type_grid', type=str, default=None, choices=['linear'], help="String : linear grid")
    parser.add_argument('-fuse','--fuse_block',type=int, default=7, choices=[7,8])
    parser.add_argument('-freq','--freq_fuse_type',type=str, nargs='+',default=None, choices=["enc","dec"])
    parser.add_argument('-fno','--use_fno',action="store_true", default=False)
    # CHANGE 1: Added arguments for wandb and intervals
    parser.add_argument('--eval_interval', type=int, default=50, help='Interval for evaluation')
    parser.add_argument('--save_interval', type=int, default=50, help='Interval for saving model')
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Enable wandb logging')
    parser.add_argument('--project_name', type=str, default="DIF-net-baseline-vs-propose", help="wandb project name")
   
    args = parser.parse_args()
 
    ############ <BEGIN> : SMART VALIDATION ##########
    # CHANGE: Added smart validation and auto-fill for stage configurations
    stage_args = [
        "trunc_mode_stages",
        "use_sobel_stages",
        "patch_based_stages",
        "patch_size_stages",
        "factorize_mode_stages",
        "use_attn_stages"
    ]
    # Assert that parameters are not None
    assert args.trunc_mode_stages is not None, "trunc_mode_stages cannot be None"
    assert args.use_sobel_stages is not None, "use_sobel_stages cannot be None"
    assert args.patch_based_stages is not None, "patch_based_stages cannot be None"
    assert args.patch_size_stages is not None, "patch_size_stages cannot be None"
    assert args.factorize_mode_stages is not None, "factorize_mode_stages cannot be None"
    assert args.use_attn_stages is not None, "use_attn_stages cannot be None"
 
    # Validate lengths of lists with more than one element
    multi_element_lengths = [len(getattr(args, arg_name)) for arg_name in stage_args if len(getattr(args, arg_name)) > 1]
    print(multi_element_lengths)
    if multi_element_lengths:
        if len(set(multi_element_lengths)) > 1:
            parser.error(f"Inconsistent lengths for stage arguments with multiple elements: {multi_element_lengths}")
        num_stages = multi_element_lengths[0]
    else:
        # All stage args have single element - default to 4 stages (matching UNet's 4 down blocks)
        num_stages = 4
    # Auto-fill single-element lists
    for arg_name in stage_args:
        arg_value = getattr(args, arg_name)
        if len(arg_value) == 1:
            setattr(args, arg_name, arg_value * num_stages)
    # Validate all lists have correct length
    for arg_name in stage_args:
        arg_value = getattr(args, arg_name)
        if len(arg_value) != num_stages:
            parser.error(f"{arg_name} length ({len(arg_value)}) must match number of stages ({num_stages})")
    # Ensure at least one of patch_based or trunc_mode is enabled for each stage
    for i in range(num_stages):
        if not (args.patch_based_stages[i] or args.trunc_mode_stages[i]):
            parser.error(f"Stage {i}: At least one of patch_based_stages[{i}] or trunc_mode_stages[{i}] must be True")
    ############ <END> : SMART VALIDATION ##########
 
    # Dynamically generate args.name if not provided
    if args.name is None:
        args.name = f"DIF-net-{args.setting}-{args.dst_list}-{args.num_views}v"
    print(args)
 
    save_dir = os.path.join(args.logdir,f'{args.name}')
    os.makedirs(save_dir, exist_ok=True)
    dst_root = args.dst_root
 
    # CHANGE 1: Save args to JSON file
    time_str = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    with open(os.path.join(save_dir, f'args_{time_str}.json'), 'w') as f:
        args_json = vars(args)
        json.dump(args_json, f, indent=4)
    # CHANGE 1: Save Git info
    git_info = get_git_info()
    with open(os.path.join(save_dir, f'git_info_{time_str}.json'), 'w') as f:
        json.dump(git_info, f, indent=4)
    print(f"Saved Git info to {os.path.join(save_dir, f'git_info_{time_str}.json')}")
 
    # CHANGE 1: Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            name=args.name,
            dir="logs_wandb",
            config={"args": vars(args)}
        )
 
    # -- initialize training dataset/loader
    dst_list = args.dst_list.split('+')
    train_dst = Mixed_CBCT_dataset(
        dst_list=dst_list,
        dst_root=dst_root,
        split='train',
        num_views=args.num_views,
        npoint=args.num_points,
        random_views=args.random_views,
    )
    train_loader = DataLoader(
        train_dst,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
 
    # -- initialize evaluation dataset/loader
    eval_loader = DataLoader(
        Mixed_CBCT_dataset(
            dst_list=dst_list,
            dst_root=dst_root,
            split='eval',
            num_views=args.num_views,
            # out_res=128, # low-res evaluation is faster
        ),
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )
 
    # -- initialize model
    model = DIF_Net(
        num_views=args.num_views,
        combine=args.combine,
        trunc_mode_stages=args.trunc_mode_stages,
        use_sobel_stages=args.use_sobel_stages,
        patch_based_stages=args.patch_based_stages,
        patch_size_stages=args.patch_size_stages,
        factorize_mode_stages=args.factorize_mode_stages,
        use_attn_stages=args.use_attn_stages,
        type_grid=args.type_grid,
        fuse_block=args.fuse_block,
        freq_fuse_type=args.freq_fuse_type,
        use_fno=args.use_fno
    ).cuda()
 
    if args.use_fno:
        display_model_info(model.image_encoder.unet_freq.get_model_info(verbose=0), save_path=os.path.join(save_dir, f'model_info_verbose0.txt'))
        display_model_info(model.image_encoder.unet_freq.get_model_info(verbose=2), save_path=os.path.join(save_dir, f'model_info_verbose2.txt'))
 
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total Parameters: {total_params}')
   
    # -- initialize optimizer, lr scheduler, and loss function
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.98,
        weight_decay=1e-3
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=np.power(0.001, 1 / args.epoch)
    )
    loss_func = nn.MSELoss()
 
    # CHANGE 2: Load checkpoint if --resume is specified
    start_epoch = 0  # Default starting epoch
    if args.resume is not None:
        checkpoint_path = os.path.join(save_dir, f'ep_{args.resume}.pth')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            start_epoch = args.resume + 1  # Resume from the next epoch
            print(f"Resumed training from checkpoint: {checkpoint_path}, starting at epoch {start_epoch}")
            # Sync the learning rate scheduler to the resumed epoch
            lr_scheduler.last_epoch = args.resume  # Set scheduler to the correct epoch
        else:
            print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")

    # CHANGE: Initialize best metrics tracking
    best_psnr = -1.0
    best_ssim = -1.0
    best_psnr_path = None
    best_ssim_path = None
 
    # -- training starts
    for epoch in tqdm(range(start_epoch,args.epoch + 1), desc="Epochs"):
        loss_list = []
        model.train()
 
        for item in tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False):
            optimizer.zero_grad()
 
            item = convert_cuda(item)
            pred, gt = model(item)
 
            loss = loss_func(pred, gt)
            loss_list.append(loss.item())
 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
 
        # -- log loss
        if epoch % 5 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch, np.mean(loss_list)))
            if args.use_wandb:
                wandb.log({"train/loss": loss, "epoch": epoch})
       
        # -- save ckpt
        if epoch % args.save_interval == 0 or (epoch >= (args.epoch - 100) and epoch % 20 == 0):
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f'ep_{epoch}.pth')
            )
 
        # -- evaluation
        if epoch % args.eval_interval == 0 or (epoch >= (args.epoch - 100) and epoch % 20 == 0):
            metrics, _ = eval_one_epoch(
                model,
                eval_loader,
                args.eval_npoint,
                use_tqdm=True
            )
            msg = f' --- epoch {epoch}'
            wandb_metrics = {"epoch": epoch}
            
            # CHANGE: Track best metrics across all datasets
            epoch_avg_psnr = 0.0
            epoch_avg_ssim = 0.0
            num_datasets = 0
            
            if metrics:  # Check if metrics is non-empty
                for dst_name in metrics.keys():
                    msg += f', {dst_name}'
                    met = metrics[dst_name]
                    for key, val in met.items():
                        msg += ', {}: {:.4}'.format(key, val)
                        wandb_metrics[f"eval/{dst_name}/{key}"] = val
                    
                    # CHANGE: Accumulate metrics for average calculation
                    epoch_avg_psnr += met['psnr']
                    epoch_avg_ssim += met['ssim']
                    num_datasets += 1
                
                # CHANGE: Calculate average metrics across all datasets
                if num_datasets > 0:
                    epoch_avg_psnr /= num_datasets
                    epoch_avg_ssim /= num_datasets
                    
                    # CHANGE: Check if this epoch has best PSNR
                    if epoch_avg_psnr > best_psnr:
                        best_psnr = epoch_avg_psnr
                        # Remove old best PSNR checkpoint
                        if best_psnr_path and os.path.exists(best_psnr_path):
                            os.remove(best_psnr_path)
                        # Save new best PSNR checkpoint
                        best_psnr_path = os.path.join(save_dir, f'ep_{epoch}_best_PSNR_{best_psnr:.4f}_SSIM_{epoch_avg_ssim:.4f}.pth')
                        torch.save(model.state_dict(), best_psnr_path)
                        print(f"New best PSNR: {best_psnr:.4f}, saved to {best_psnr_path}")
                    
                    # CHANGE: Check if this epoch has best SSIM
                    if epoch_avg_ssim > best_ssim:
                        best_ssim = epoch_avg_ssim
                        # Remove old best SSIM checkpoint
                        if best_ssim_path and os.path.exists(best_ssim_path):
                            os.remove(best_ssim_path)
                        # Save new best SSIM checkpoint
                        best_ssim_path = os.path.join(save_dir, f'ep_{epoch}_PSNR_{epoch_avg_psnr:.4f}_best_SSIM_{best_ssim:.4f}.pth')
                        torch.save(model.state_dict(), best_ssim_path)
                        print(f"New best SSIM: {best_ssim:.4f}, saved to {best_ssim_path}")
            
            print(msg)
            if args.use_wandb:
                wandb.log(wandb_metrics)
       
        lr_scheduler.step()