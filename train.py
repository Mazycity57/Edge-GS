#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
import os
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

import torchvision.transforms as T
import torch.nn.functional as F
import math
import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.unet_model import UNet
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    
    from torch.utils.tensorboard import SummaryWriter # 使用PyTorch自带的tensorboard
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
# dataset: 数据集对象，包含训练数据。
# opt: 训练选项，包含训练配置参数。
# pipe: 渲染管道对象。
# dataset_name: 数据集名称。
# testing_iterations: 测试迭代次数。
# saving_iterations: 保存模型的迭代次数。
# checkpoint_iterations: 保存检查点的迭代次数。
# checkpoint: 检查点文件路径。
# debug_from: 启用调试模式的迭代次数。
# wandb: 用于日志记录的Weights and Biases对象。
# logger: 日志记录器。
# ply_path: PLY文件路径，用于存储点云数据。
    
    #初始化，数据集图片在初始化阶段不需要被读入，而是在训练过程的循环中被动态加载和处理。
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)#用于TensorBoard日志记录。
    # import pdb
    # pdb.set_trace()
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)#update_depth定义了锚点更新的迭代深度，update_hierachy_factor用于调整锚点更新过程中阈值和大小因子，使得锚点的更新逐渐精细化。
    # pdb.set_trace()
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)#场景对象，包含训练数据和高斯模型
    gaussians.training_setup(opt)#训练设置
    if checkpoint:#加载检查点
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)#恢复模型的状态或加载预训练模型

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    # Initialize SAM model
    sam = sam_model_registry["vit_h"](checkpoint="/home/wmy/proj/Scaffold-GS-main/model/sam_vit_h_4b8939.pth").to("cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    #对数据集里面的图片进行预分割处理
    Image_pre_segmentation(scene,500,dataset.source_path)
    
    viewpoint_stack = None  # 初始化为空列表
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")#使用了 tqdm 库来创建一个带有进度条的迭代器，以显示训练过程中的进度。
    first_iter += 1

    #训练循环
    #检查并连接网络gui
    prev_seg_loss = None
    prev_ssim_loss=None
    prev_L1_loss =None
    # 定义一个变量来记录最小损失
    best_loss = float('inf')
    best_iteration = -1  # 可选：记录最小损失发生的迭代次数
    for iteration in range(first_iter, opt.iterations + 1):        
        # network gui not available in scaffold-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)#更新学习率
        #设置背景颜色
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # 从训练相机堆栈中随机选择一个相机
         # Pick a random Camera，从训练相机堆栈中随机选择一个相机。
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()#如果相机堆栈为空重新填充堆栈
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))#从相机堆栈中随机选择一个相机 viewpoint_cam。这个相机的视角会被用于渲染和计算损失。

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)#预过滤体素
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)#渲染过程的输出包
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
     
        gt_image = viewpoint_cam.original_image.cuda()
            # import pdb
            # pdb.set_trace()
        # # 将 gt_image 从 PyTorch Tensor 转换为 NumPy 数组
        # gt_image_np = gt_image.detach().cpu().numpy()
    
        # # 在调用前转换 gt_image_np
        # if gt_image_np.shape[0] == 3:
        #    gt_image_np = np.transpose(gt_image_np, (1, 2, 0))  # 转换为 (H, W, C)
        #    gt_image_np = cv2.cvtColor(gt_image_np, cv2.COLOR_RGB2GRAY)  # 转换为灰度图

        # # 调用 gradient_based_edge_indicator 时使用 NumPy 数组
        # edge_weights = gradient_based_edge_indicator(gt_image_np, beta=0.5, p=2)

        # # Ll1 损失仍然使用 PyTorch Tensor 的计算
        # Ll1 =  (edge_weights * l1_loss(image, gt_image)).mean()
        
        
        Ll1 =l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))#亮度、对比度和结构三方面的比较
        scaling_reg = scaling.prod(dim=1).mean()#对每个点的缩放因子进行乘积，用于防止缩放因子过大或过小，以维持渲染过程中合理的缩放范围，从而提升模型的稳定性。
        # print(f"Iteration {iteration}, Ll1: {Ll1},ssim_loss: {ssim_loss}")
        
        if should_compute_seg_loss(iteration, Ll1, ssim_loss, prev_L1_loss, prev_ssim_loss):
    
            # 对渲染后的图像进行SAM分割
            sam_input_image = (image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            seg_masks_image_list = mask_generator.generate(sam_input_image)
            image_name = viewpoint_cam.image_name
            seg_path = os.path.join(dataset.source_path, 'seg_results')
            segmentation_file_path = os.path.join(seg_path, f'segmentation_{image_name}.npy')
            gt_segmentation = np.load(segmentation_file_path) 
            
             
            # 合并所有面积较大的物体的分割掩码
            min_area = 500  # 设置面积阈值，可根据需要调整
            combined_mask = None  # 初始化 combined_mask
    
            if len(seg_masks_image_list) > 0:
                for mask_dict in seg_masks_image_list:
                    mask = mask_dict["segmentation"]
                    area = np.sum(mask)  # 计算当前掩码的面积
                    if area >= min_area:  # 仅保留面积大于 min_area 的掩码
                        if combined_mask is None:
                            combined_mask = mask.astype(np.uint8)  # 初始化综合掩码
                        else:
                            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)

                # 确保 combined_mask 已定义并调用 visualize 函数
                if combined_mask is not None:
                    # if iteration % 1000 == 0:
                        visualize(iteration, dataset_name,image, gt_image, gt_segmentation, combined_mask)
                else:
                    print(f"Warning: combined_mask is None for iteration {iteration}, skipping visualization.")

                # 处理分割结果
                if gt_segmentation is None:
                    print(f"Warning: No segmentation generated for iteration {iteration}.")
                    seg_loss = 0
                else:
                    # 计算分割损失
                    seg_loss, prev_seg_loss = compute_segmentation_loss(combined_mask, gt_segmentation, prev_seg_loss)
                    tb_writer.add_scalar('Segmentation Loss', seg_loss, iteration)
                    print(f"Iteration {iteration},  Segmentation Loss: {seg_loss}, Previous Segmentation Loss: {prev_seg_loss}")

                    # 更新前一个分割损失
                    prev_seg_loss = seg_loss

            else:
                seg_loss = 0  # 没有生成分割的情况
        else:
            seg_loss = 0  # 没有计算时设为 0
        
        prev_L1_loss=Ll1
        prev_ssim_loss=ssim_loss
        # Combine losses
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg + opt.lambda_seg * seg_loss   
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg
        loss.backward()
        
        iter_end.record()

        #记录和保存
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss,seg_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            # training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            # if (iteration in saving_iterations):
            #     logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)
            
            if loss.item() < best_loss:  # 如果当前损失小于最小损失
                best_loss = loss.item()  # 更新最小损失
                best_iteration = iteration  # 更新最小损失的迭代次数
                logger.info(f"[ITER {iteration}] Found new best model with loss: {best_loss}")


            # densification，调整锚点
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                     # 在渲染后计算图像的梯度
                    gradient_magnitude = compute_sobel_gradient(gt_image)
                    gaussians.adjust_anchor(check_interval=opt.update_interval, 
                                            success_threshold=opt.success_threshold,
                                            grad_threshold=opt.densify_grad_threshold,
                                            min_opacity=opt.min_opacity,
                                            gradient_magnitude = gradient_magnitude 
                                            )
                    # # 获取点云数量
                    # point_count = gaussians._anchor.size(0)  # _anchor 的第一个维度表示点的数量
                    # # 打印当前点云数量
                    # print(f"Epoch {iteration}, Point Count: {point_count}")
        
                    # # 在渲染后计算图像的梯度
                    # gradient_magnitude = compute_sobel_gradient(gt_image)
                    # if gradient_magnitude is None:
                    #     print("Error: gradient_magnitude is None")
                    # else:
                    #     print(f"Gradient Magnitude Shape: {gradient_magnitude.shape}")
                    #     print(f"Min: {gradient_magnitude.min()}, Max: {gradient_magnitude.max()}")
                    # # 使用梯度信息来增长锚点
                    # gaussians.readjust_anchor(check_interval=opt.update_interval, 
                    #                         success_threshold=opt.success_threshold,
                    #                         grad_threshold=opt.densify_grad_threshold,
                    #                         min_opacity=opt.min_opacity,
                    #                         gradient_magnitude=gradient_magnitude)
                    
                    # point_count1 = gaussians._anchor.size(0) 
                    # print(f"Epoch {iteration}, Point Count1: {point_count1}")
        
                 
                    

            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    # logger.info(f"\n[ITER {best_iteration}] Saving Gaussians with best loss {best_loss}")
    scene.save(best_iteration)  # 仅保存最优损失对应的迭代信息
                

def visualize(iteration,dataset_name,image, gt_image, gt_segmentation,seg_masks_image):
   # 可视化
    # 将张量从 GPU 移动到 CPU，并转为 Numpy 数组
    
    image_np = image.detach().cpu().numpy()
    gt_image_np = gt_image.detach().cpu().numpy()
    # downsampled_gt_image_np = downsampled_gt_image.detach().cpu().numpy()
    gt_segmentation_np=gt_segmentation
    seg_masks_image_np=seg_masks_image

    # 转换张量的形状从 (C, H, W) 到 (H, W, C)
    image_np = image_np.transpose(1, 2, 0)
    gt_image_np = gt_image_np.transpose(1, 2, 0)
    # downsampled_gt_image_np = downsampled_gt_image_np.transpose(1, 2, 0)
    # 检查 gt_segmentation_np 的形状并进行转置
    if gt_segmentation_np.ndim == 3:  # 如果是 (C, H, W)
        gt_segmentation_np = gt_segmentation_np.transpose(1, 2, 0)
    elif gt_segmentation_np.ndim == 2:  # 如果是 (H, W)
        gt_segmentation_np = gt_segmentation_np[np.newaxis, :, :]  # 添加通道维度
        # 检查 gt_segmentation_np 的形状并进行转置
    if seg_masks_image_np.ndim == 3:  # 如果是 (C, H, W)
        seg_masks_image_np = seg_masks_image_np.transpose(1, 2, 0)
    elif seg_masks_image_np.ndim == 2:  # 如果是 (H, W)
        seg_masks_image_np = seg_masks_image_np[np.newaxis, :, :]  # 添加通道维度

    # 将张量值从 [0, 1] 范围转换到 [0, 255] 并转换为 uint8 类型
    image_np = (image_np * 255).astype('uint8')
    gt_image_np = (gt_image_np * 255).astype('uint8')
    # downsampled_gt_image_np = (downsampled_gt_image_np * 255).astype('uint8')
    gt_segmentation_np=(gt_segmentation_np*255).astype('uint8')
    image_segmentation_np=(seg_masks_image_np*255).astype('uint8')

    # 使用 PIL 保存为 PNG 格式
    image_pil = Image.fromarray(image_np)
    gt_image_pil = Image.fromarray(gt_image_np)
    # downsampled_gt_image_pil = Image.fromarray(downsampled_gt_image_np)
    gt_segmentation_pil = Image.fromarray(gt_segmentation_np[0])  # 从 (1, H, W) 转换为 (H, W)
    image_segmentation_pil = Image.fromarray(image_segmentation_np[0])
    # 保存到指定路径
        
    save_dir = f'/home/wmy/proj/Scaffold-GS-main/outputs/{dataset_name}'
    os.makedirs(save_dir, exist_ok=True)  # 如果路径不存在则创建

     # 保存图片
    try:
        image_pil.save(os.path.join(save_dir, f'{dataset_name}_rendered_image_{iteration}.png'))
        gt_image_pil.save(os.path.join(save_dir, f'{dataset_name}_gt_image_{iteration}.png'))
        # downsampled_gt_image_pil.save(os.path.join(save_dir, 'downsampled_gt_image.png'))
        gt_segmentation_pil.save(os.path.join(save_dir, f'{dataset_name}_seg_gt_{iteration}.png'))
        image_segmentation_pil.save(os.path.join(save_dir, f'{dataset_name}_seg_rendered_{iteration}.png'))
    except Exception as e:
        print(f"Error saving images: {e}")
            


def should_compute_seg_loss(iteration, L1_loss, ssim_loss, 
                                prev_L1_loss, prev_ssim_loss,
                             threshold=0.1, threshold_ssim=0.2, ):
        # 动态调整阈值
        dynamic_threshold = threshold * (1 - iteration / 30000)
        dynamic_threshold_ssim = threshold_ssim * (1 - iteration / 30000)
    
        # 初期减少频率
        if iteration < 10000:
            return iteration % 5000 == 0
        # 中期平稳提升频率
        elif iteration < 20000:
            return iteration % 1000 == 0
        # 后期频率动态调整
        else:
            # 如果 L1_loss 或 ssim_loss 变化大，增加计算频率
            if L1_loss is not None and (prev_L1_loss is not None and abs(L1_loss - prev_L1_loss) > dynamic_threshold):
                return iteration % 100 == 0
            elif ssim_loss is not None and (prev_ssim_loss is not None and abs(ssim_loss - prev_ssim_loss) > dynamic_threshold_ssim):
                return iteration % 100 == 0
            else:
                return iteration % 200 == 0  # 正常频率
            
def Image_pre_segmentation(scene, min_area, dataset_path):  # 增加 min_area 参数 
    # Initialize SAM model
    sam = sam_model_registry["vit_h"](checkpoint="/home/wmy/proj/Scaffold-GS-main/model/sam_vit_h_4b8939.pth").to("cuda")
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    viewpoint_stack = []  # 初始化为空列表
    gt_data = []  # 存储 gt_image 和其对应的降采样与分割结果
    
    # 预处理所有 gt_image
    save_dir = os.path.join(dataset_path, "seg_results")  # 根据 dataset_path 动态生成保存路径
    os.makedirs(save_dir, exist_ok=True)  # 创建目录，如果已经存在则不会报错
    cameras = scene.getTrainCameras().copy()
    selected_indices = []  # 存储选中的相机索引

    for i, cam in enumerate(cameras):
        gt_image = cam.original_image.cuda()
        sam_input_gt_image = (gt_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # 尝试加载已保存的分割结果
        seg_masks_path = os.path.join(save_dir, f'segmentation_{cam.image_name}.npy')  
        if os.path.exists(seg_masks_path):
            seg_masks_gt = np.load(seg_masks_path)  # 加载存储的分割结果
        else:
            # 生成分割结果
            seg_masks_gt_list = mask_generator.generate(sam_input_gt_image)
            print(f"Generated masks for {cam.image_name}: {len(seg_masks_gt_list)} masks found.")
            
            # 合并面积较大的物体的分割掩码
            seg_masks_gt = None
            valid_masks_count = 0  # 记录有效掩码的数量
            for mask_dict in seg_masks_gt_list:
                mask = mask_dict["segmentation"]
                area = np.sum(mask)  # 计算掩码的像素面积
                if area >= min_area:  # 仅保留大于 min_area 的物体
                    valid_masks_count += 1  # 有效掩码数量加1
                    if seg_masks_gt is None:
                        seg_masks_gt = mask.astype(np.uint8)  # 初始化综合掩码
                    else:
                        seg_masks_gt = np.logical_or(seg_masks_gt, mask).astype(np.uint8)

            # 输出调试信息并存储分割结果
            if seg_masks_gt is not None:
                print(f"Segmentation succeeded for image: {cam.image_name}, saving to {seg_masks_path}. Found {valid_masks_count} large masks.")
                np.save(seg_masks_path, seg_masks_gt)  # 保存为 .npy 文件
            else:
                print(f"No large enough segmentation found for image: {cam.image_name}")
                seg_masks_gt = None

        # 将当前相机和分割结果添加到列表中
        viewpoint_stack.append(cam)
        gt_data.append({
            'gt_image': gt_image,  # 存储原始分辨率的 GT 图像
            'segmentation': seg_masks_gt  # 存储合并后的分割掩码
        })
        selected_indices.append(i)  # 记录选中的相机索引
    
    # 过滤出有效的分割结果，返回包含非空分割的 gt_data
    filtered_gt_data = [data for data in gt_data if data['segmentation'] is not None]

    return viewpoint_stack, filtered_gt_data, selected_indices

       
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
#     if tb_writer:
#         tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
#         tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
#         tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


#     if wandb is not None:
#         wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
#     # Report test and samples of training set
#     if iteration in testing_iterations:
#         scene.gaussians.eval()
#         torch.cuda.empty_cache()
#         validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
#                               {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

#         for config in validation_configs:
#             if config['cameras'] and len(config['cameras']) > 0:
#                 l1_test = 0.0
#                 psnr_test = 0.0
                
#                 if wandb is not None:
#                     gt_image_list = []
#                     render_image_list = []
#                     errormap_list = []

#                 for idx, viewpoint in enumerate(config['cameras']):
#                     voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
#                     image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
#                     gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
#                     if tb_writer and (idx < 30):
#                         tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
#                         tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

#                         if wandb:
#                             render_image_list.append(image[None])
#                             errormap_list.append((gt_image[None]-image[None]).abs())
                            
#                         if iteration == testing_iterations[0]:
#                             tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
#                             if wandb:
#                                 gt_image_list.append(gt_image[None])

#                     l1_test += l1_loss(image, gt_image).mean().double()
#                     psnr_test += psnr(image, gt_image).mean().double()

                
                
#                 psnr_test /= len(config['cameras'])
#                 l1_test /= len(config['cameras'])          
#                 logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
#                 if tb_writer:
#                     tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
#                     tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
#                 if wandb is not None:
#                     wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

#         if tb_writer:
#             # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
#             tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
#         torch.cuda.empty_cache()

#         scene.gaussians.train()

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, seg_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/seg_loss', seg_loss.item() if isinstance(seg_loss, torch.Tensor) else seg_loss, iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)
        
    if wandb is not None:
        wandb.log({
            "train_l1_loss": Ll1.item(),
            "train_total_loss": loss.item(),
            "train_seg_loss": seg_loss.item() if seg_loss is not None else 0
        })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({
            'name': 'test', 
            'cameras': scene.getTestCameras()
        }, {
            'name': 'train', 
            'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]
        })

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    tb_writer.add_images(f'{dataset_name}/{config["name"]}_view_{viewpoint.image_name}/render', image[None], global_step=iteration)
                    tb_writer.add_images(f'{dataset_name}/{config["name"]}_view_{viewpoint.image_name}/errormap', (gt_image[None] - image[None]).abs(), global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(f'{dataset_name}/{config["name"]}_view_{viewpoint.image_name}/ground_truth', gt_image[None], global_step=iteration)

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                tb_writer.add_scalar(f'{dataset_name}/{config["name"]}/loss_viewpoint_l1_loss', l1_test, iteration)
                tb_writer.add_scalar(f'{dataset_name}/{config["name"]}/loss_viewpoint_psnr', psnr_test, iteration)
                
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss": l1_test, f"{config['name']}_PSNR": psnr_test})

        tb_writer.add_scalar(f'{dataset_name}/total_points', scene.gaussians.get_anchor.shape[0], iteration)

        torch.cuda.empty_cache()
        scene.gaussians.train()


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


def compute_canny_edge(image, low_threshold=100, high_threshold=200):
    # 如果是4D或3D图像，先转为灰度图
    if image.dim() == 4:
        image = torch.mean(image, dim=1, keepdim=True)
    elif image.dim() == 3:
        image = torch.mean(image, dim=0, keepdim=True).unsqueeze(0)
    elif image.dim() != 2:
        raise ValueError(f"Expected 2D, 3D, or 4D input, but got {image.dim()}D input.")
    
    # 将Torch张量转为NumPy数组
    image_np = image.squeeze().cpu().numpy().astype('uint8')
    
    # 使用OpenCV Canny边缘检测
    edges = cv2.Canny(image_np, low_threshold, high_threshold)
    
    # 将边缘图转回Torch张量
    edges_tensor = torch.tensor(edges, dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    
    return edges_tensor.squeeze()


def compute_scharr_gradient(image):
    if image.dim() == 4:
        image = torch.mean(image, dim=1, keepdim=True)
    elif image.dim() == 3:
        image = torch.mean(image, dim=0, keepdim=True).unsqueeze(0)
    elif image.dim() != 2:
        raise ValueError(f"Expected 2D, 3D, or 4D input, but got {image.dim()}D input.")

    scharr_x = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    scharr_y = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)

    gradient_x = torch.nn.functional.conv2d(image, scharr_x, padding=1)
    gradient_y = torch.nn.functional.conv2d(image, scharr_y, padding=1)
    scharr_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
    return scharr_magnitude.squeeze()

def compute_sobel_gradient(image):
    """
    计算图像的梯度。
    使用 Sobel 滤波器计算梯度。
    """
    # 将图像转换为灰度图，支持 3D 或 4D 输入
    if image.dim() == 4:
        image = torch.mean(image, dim=1, keepdim=True) # [B, 1, H, W]
    elif image.dim() == 3:
        image = torch.mean(image, dim=0, keepdim=True).unsqueeze(0) # [1, 1, H, W]
    elif image.dim() != 2:
        raise ValueError(f"Expected 2D, 3D, or 4D input, but got {image.dim()}D input.")
 
    # 定义 Sobel 滤波器
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)

    # 计算 x 和 y 方向的梯度
    gradient_x = torch.nn.functional.conv2d(image, sobel_x, padding=1)
    gradient_y = torch.nn.functional.conv2d(image, sobel_y, padding=1)
    sobel_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
    return sobel_magnitude.squeeze()


def compute_segmentation_loss(pred_mask, gt_mask, prev_loss, contour_weight=0.3, iteration=0, total_iterations=30000, alpha=0.9):
    """
    计算分割损失，动态调整轮廓损失和高斯平滑核大小。
    """

    # 如果输入是 numpy 数组，转换为 torch 张量
    if isinstance(pred_mask, np.ndarray): 
        pred_mask = torch.tensor(pred_mask, dtype=torch.float32).to("cuda") 
    if isinstance(gt_mask, np.ndarray): 
        gt_mask = torch.tensor(gt_mask, dtype=torch.float32).to("cuda") 

    # 计算 Soft Dice 损失
    intersection = (pred_mask * gt_mask).sum()
    dice_loss = 1 - (2. * intersection + 1e-6) / (pred_mask.sum() + gt_mask.sum() + 1e-6)

    # 动态调整轮廓损失的权重，增加条件判断，使权重在一定范围内波动
    if iteration > 0.5 * total_iterations:
        dynamic_contour_weight = max(0.05, contour_weight * (1 - iteration / total_iterations))
    else:
        dynamic_contour_weight = contour_weight

    # # 动态调整高斯平滑的核大小
    # kernel_size = max(3, int(7 - 4 * (iteration / total_iterations)))
    # if kernel_size % 2 == 0:
    #     kernel_size += 1  # kernel_size 必须为奇数
    
    # 动态调整高斯平滑核大小
    kernel_size = max(3, int(7 - 4 * (iteration / total_iterations)))
    kernel_size = kernel_size if iteration < 0.7 * total_iterations else max(3, kernel_size - 2)
    
    # 确保 kernel_size 为奇数
    if kernel_size % 2 == 0:
        kernel_size += 1



    # 应用高斯平滑，减少噪声影响
    pred_mask_np = cv2.GaussianBlur(pred_mask.detach().cpu().numpy().astype(np.float32), (kernel_size, kernel_size), 0)
    gt_mask_np = cv2.GaussianBlur(gt_mask.detach().cpu().numpy().astype(np.float32), (kernel_size, kernel_size), 0)

    # 转换为 uint8 格式的灰度图
    pred_mask_np = (pred_mask_np * 255).astype(np.uint8)
    gt_mask_np = (gt_mask_np * 255).astype(np.uint8)
    pred_mask_np = pred_mask_np.squeeze()
    gt_mask_np = gt_mask_np.squeeze()

    # 确保输入为单通道灰度图像
    if pred_mask_np.ndim == 3:
        pred_mask_np = cv2.cvtColor(pred_mask_np, cv2.COLOR_RGB2GRAY)
    if gt_mask_np.ndim == 3:
        gt_mask_np = cv2.cvtColor(gt_mask_np, cv2.COLOR_RGB2GRAY)

    # 使用 HOG 特征提取边缘信息
    hog = cv2.HOGDescriptor()
    pred_contours = hog.compute(pred_mask_np)
    gt_contours = hog.compute(gt_mask_np)

    # 转换为 torch 张量
    pred_contours = torch.tensor(pred_contours, dtype=torch.float32).to("cuda")
    gt_contours = torch.tensor(gt_contours, dtype=torch.float32).to("cuda")

    # 计算 L2 损失（也可以尝试其他损失，如 SSIM）
    contour_loss = F.mse_loss(pred_contours, gt_contours)

    # 总损失 = Soft Dice 损失 + 动态轮廓损失
    total_loss = dice_loss + dynamic_contour_weight * contour_loss

    # 添加动态平滑：如果损失下降较慢，减小 alpha，增加当前损失的权重
    if prev_loss is not None:
        if total_loss > prev_loss:
            alpha = max(0.6, alpha - 0.05)  # 如果损失上升，减少平滑效果
        else:
            alpha = min(0.95, alpha + 0.05)  # 如果损失下降，增加平滑效果
        total_loss = alpha * prev_loss + (1 - alpha) * total_loss

    return total_loss,  prev_loss  # 返回当前损失和上次使用的损失





if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    

    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
    
    dataset = args.source_path.split('/')[-1]#提取数据集
    exp_name = args.model_path.split('/')[-2]#提取实验名称
    print(dataset)
    if args.use_wandb:#如果启用了 use_wandb，则登录并初始化一个 WandB 运行，记录超参数和运行元数据。
        wandb.login()#WandB 是一个实验跟踪和协作工具，用于记录和可视化训练过程。
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)#优化日志记录

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:#Warmup 是一种训练技术，通过在训练初期使用较低的学习率并逐步增加，以确保训练过程的稳定性和防止梯度爆炸。
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
