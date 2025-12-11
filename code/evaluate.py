import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import csv

import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import Mixed_CBCT_dataset
from models.model import DIF_Net
from utils import convert_cuda, add_argument, save_nifti


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
def eval_one_epoch(model, loader, npoint=50000, save_dir=None, ignore_msg=True, use_tqdm=False):
    model.eval()
    results = {}
    metrics = {}
    metrics_tmp = {key:[] for key in ['psnr', 'ssim']} # , 'rmse', 'mse', 'mae']}
    if use_tqdm:
        loader = tqdm(loader, ncols=50)
    
    with torch.no_grad():
        for item in loader:
            item = convert_cuda(item)

            dst_name = item['dst_name'][0]
            name = item['name'][0]
            image = item['p_gt'].cpu().numpy() # [1, W, H, D]
            image = image[0] # W, H, D

            output = model(item, is_eval=True, eval_npoint=npoint) # B, 1, N
            output = output[0, 0].data.cpu().numpy()
            output = output.reshape(image.shape)
            output = np.clip(output, 0, 1)
            psnr = peak_signal_noise_ratio(image, output, data_range=1.)
            ssim = structural_similarity(image, output, data_range=1.)

            if not ignore_msg:
                print('{}, PSNR: {:.4}, SSIM: {:.4}'.format(
                    name, psnr, ssim
                ))

            dst_res = results.get(dst_name, [])
            dst_met = metrics.get(dst_name, deepcopy(metrics_tmp))

            dst_res.append({
                'name': name, 
                'psnr': psnr,
                'ssim': ssim,
            })
            for key in dst_met.keys():
                dst_met[key].append(dst_res[-1][key])
            
            results[dst_name] = dst_res
            metrics[dst_name] = dst_met

            if save_dir is not None:
                output = np.clip(output, 0, 1)
                output *= 255.
                output = output.astype(np.uint8)
                save_path = os.path.join(save_dir, f'{name}.nii.gz')
                save_nifti(output, save_path)

    for dst_name in metrics.keys():
        dst_met = metrics[dst_name]
        m = {key:np.mean(val) for key, val in dst_met.items()}
        metrics[dst_name] = m
    
    return metrics, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval')
    parser = add_argument(parser, train=False)
    # Additional params
    parser.add_argument('-trunc','--trunc_mode_stages', type=parse_none,nargs='+', default=None, help="Comma-separated list of trunc modes (e.g., 'LL-LH,LH-HH,shared_sliding') or 'None'")
    parser.add_argument('-sobel','--use_sobel_stages', type=str_to_bool, nargs='+', default=None, help="List of booleans (e.g., 'True False True') or 'None'")
    parser.add_argument('-patch','--patch_based_stages', type=str_to_bool, nargs='+', default=None, help="List of booleans (e.g., 'True False True') or 'None'")
    parser.add_argument('-psize','--patch_size_stages', type=str_to_int, default=None,nargs='+', help="Patch size as an integer, or None")
    parser.add_argument('-fac','--factorize_mode_stages', type=parse_none,nargs='+', default=None, help="Comma-separated list of factorize modes (e.g., 'mode1,mode2,mode3') or 'None'")
    parser.add_argument('-attn','--use_attn_stages', type=str_to_bool, nargs='+', default=None, help="List of booleans (e.g., 'True False True') or 'None'")
    parser.add_argument('-grid','--type_grid', type=str, default=None, choices=['linear'], help="String : linear grid")
    parser.add_argument('-fuse','--fuse_block',type=int, default=7, choices=[7,8])
    parser.add_argument('-freq','--freq_fuse_type',type=str, default=None, nargs='+',choices=['enc','dec'])
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
    dst_root = args.dst_root
    # -- dataloader
    eval_loader = DataLoader(
        Mixed_CBCT_dataset(
            dst_list=args.dst_list.split('+'),
            dst_root=dst_root,
            split=args.split, 
            num_views=args.num_views,
            # out_res=args.out_res,
            view_offset=args.view_offset,
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=True
    )

    # -- model, load ckpt
    ckpt_path =  os.path.join(args.logdir,f'{args.name}/ep_{args.epoch}.pth')
    ckpt = torch.load(ckpt_path)
    print('load ckpt from', ckpt_path)
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
    )
    model.load_state_dict(ckpt)
    model = model.cuda()

    # -- output dir
    save_dir = None
    if args.visualize:
        save_dir = os.path.join(args.logdir,f'{args.name}/results/ep_{args.epoch}/predictions')
        os.makedirs(save_dir, exist_ok=True)

    # -- evaluate
    metrics, results = eval_one_epoch(
        model, 
        eval_loader, 
        args.eval_npoint,
        save_dir=save_dir,
        ignore_msg=False,
        use_tqdm=False
    )
    print(metrics)

    # -- save results
    pred_dir = os.path.join(args.logdir,f'{args.name}/results/ep_{args.epoch}')
    os.makedirs(pred_dir, exist_ok=True)

    csv_file = open(os.path.join(pred_dir, 'results.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['dataset', 'obj_id', 'psnr', 'ssim'])

    for dst_name in results.keys():
        dst_res = results[dst_name]
        for res in dst_res:
            csv_writer.writerow([dst_name, res['name'], res['psnr'], res['ssim']])

        dst_avg = metrics[dst_name]
        csv_writer.writerow([dst_name, 'average', dst_avg['psnr'], dst_avg['ssim']])
    
    csv_file.close()

    with open(os.path.join(pred_dir, 'args.json'), 'w') as f:
        args = vars(args)
        json.dump(args, f, indent=4)
