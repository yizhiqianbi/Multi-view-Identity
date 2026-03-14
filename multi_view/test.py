"""
Multi-view Video Generation Test Script

支持多种测试模式：
1. original: 使用 datasets/test_set 中的 case1-10 数据集
2. selected10: 使用 examples/selected10 数据集，支持多GPU并行
3. train_set: 在训练集上运行测试，输出原始视频和生成视频
"""

import torch
from PIL import Image, ImageOps
from einops import rearrange
import numpy as np
from typing import Optional, List, Tuple, Callable
import json
import math

from tqdm import tqdm
import os
import argparse
import torch.distributed as dist
import torch.nn.functional as F
from diffsynth.models import ModelManager
from diffsynth.models.utils import load_state_dict
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import yaml
from diffsynth.trainers.utils import wan_parser
from datasets.videodataset import MulltiShot_MultiView_Dataset


# ============================================================================
# 工具函数
# ============================================================================

def process_ref_images(ref_images: List[Image.Image], height: int, width: int) -> List[Image.Image]:
    """
    处理参考图片：保持宽高比，填充到目标尺寸

    Args:
        ref_images: 参考图片列表
        height: 目标高度
        width: 目标宽度

    Returns:
        处理后的图片列表
    """
    ref_images_new = []
    for ref_image in ref_images:
        ref_image = ref_image.convert("RGB")
        img_ratio = ref_image.width / ref_image.height
        target_ratio = width / height

        if img_ratio > target_ratio:  # 图片更宽
            new_width = width
            new_height = int(new_width / img_ratio)
        else:  # 图片更高
            new_height = height
            new_width = int(new_height * img_ratio)

        ref_image = ref_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 居中填充
        delta_w = width - ref_image.size[0]
        delta_h = height - ref_image.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        new_img = ImageOps.expand(ref_image, padding, fill=(255, 255, 255))
        ref_images_new.append(new_img)
    return ref_images_new


def load_pipeline(args, checkpoint_path: str) -> WanVideoPipeline:
    """
    加载 WanVideoPipeline

    Args:
        args: 命令行参数
        checkpoint_path: 模型checkpoint路径

    Returns:
        WanVideoPipeline 实例
    /root/paddlejob/workspace/qizipeng/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
    """
    print(f"Loading pipeline with checkpoint: {checkpoint_path}")
    pipe, vae = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=os.path.join(args.local_model_path, "Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"),
                       offload_device="cuda"),
            ModelConfig(path=os.path.join(args.local_model_path, "Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
                       offload_device="cuda"),
            ModelConfig(path=checkpoint_path, offload_device="cuda"),
        ],
        redirect_common_files=False
    )
    pipe.vae = vae
    pipe.enable_vram_management()
    return pipe


def setup_output_dirs(base_path: str) -> Tuple[str, str, str]:
    """
    创建输出目录

    Args:
        base_path: 基础输出路径

    Returns:
        (ref_images_dir, generated_video_dir, original_video_dir)
    """
    ref_dir = os.path.join(base_path, "ref_images")
    gen_dir = os.path.join(base_path, "video")
    orig_dir = os.path.join(base_path, "original_video")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(orig_dir, exist_ok=True)
    return ref_dir, gen_dir, orig_dir


def get_checkpoint_path(args) -> str:
    """
    获取checkpoint路径

    Args:
        args: 命令行参数

    Returns:
        checkpoint路径
    """
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
        return args.checkpoint_path
    return os.path.join(args.output_path, args.visual_log_project_name,
                       f"checkpoint-step-{args.infer_step}-epoch-{args.epoch_id}", "weights.safetensors")


# ============================================================================
# 测试函数
# ============================================================================

def test_on_train_set(args, gpu_id: int = 0, total_gpus: int = 1):
    """
    在训练集上运行测试，输出原始视频和生成视频

    Args:
        args: 命令行参数
        gpu_id: 当前GPU编号
        total_gpus: 总GPU数量
    """
    torch.cuda.set_device(gpu_id)

    # 加载配置
    with open(args.train_yaml, "r", encoding="utf-8") as f:
        conf_info = yaml.safe_load(f)

    checkpoint_path = get_checkpoint_path(args)
    output_path = os.path.join("./output", args.visual_log_project_name, "train_set_test")
    ref_dir, gen_dir, orig_dir = setup_output_dirs(output_path)

    print(f"[GPU {gpu_id}] Checkpoint: {checkpoint_path}")
    print(f"[GPU {gpu_id}] Output path: {output_path}")

    # 加载pipeline
    pipe = load_pipeline(args, checkpoint_path)

    # 加载训练集
    dataset = MulltiShot_MultiView_Dataset(
        dataset_base_path=args.dataset_base_path,
        resolution=(args.height, args.width),
        ref_num=args.ref_num,
        training=True  # 使用训练集
    )

    # 获取数据集信息
    train_size = len(dataset.data_train)

    # 根据GPU数量分配任务
    num_samples = getattr(args, 'num_test_samples', min(20, train_size))  # 默认测试20个样本
    all_indices = list(range(train_size))
    my_indices = [i for i in all_indices if i % total_gpus == gpu_id][:num_samples]

    print(f"[GPU {gpu_id}] Dataset size: {train_size}")
    print(f"[GPU {gpu_id}] Assigned samples: {len(my_indices)}")

    # 负面提示词
    negative_prompt = ["色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"]

    # 日志文件
    log_file = os.path.join(output_path, f"log_gpu{gpu_id}.txt")

    with open(log_file, "w", encoding="utf-8") as f_log:
        f_log.write(f"GPU {gpu_id} Test Log\n")
        f_log.write(f"=" * 50 + "\n")

        for idx, data_idx in enumerate(tqdm(my_indices, desc=f"[GPU {gpu_id}] Processing")):
            try:
                metadata = dataset[data_idx]

                # 保存参考图片
                for r_index, img in enumerate(metadata["ref_images"]):
                    ref_path = os.path.join(ref_dir, f"{data_idx}_{gpu_id}_ref{r_index}.png")
                    img.save(ref_path)

                # 生成视频
                video, _ = pipe(
                    args=args,
                    prompt=[metadata["single_caption"]],
                    ref_images=[metadata["ref_images"]],
                    negative_prompt=negative_prompt,
                    seed=42,
                    tiled=True,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    cfg_scale_face=5.0,
                    num_ref_images=metadata["ref_num"]
                )

                # 保存生成的视频
                gen_video_path = os.path.join(gen_dir, f"{data_idx}_{gpu_id}_generated.mp4")
                save_video(video, gen_video_path, fps=15, quality=10)

                # 提取并保存原始视频片段
                video_path = metadata["video_path"]
                orig_video_path = os.path.join(orig_dir, f"{data_idx}_{gpu_id}_original.mp4")
                _extract_video_segment(video_path, metadata, orig_video_path)

                # 记录日志
                log_entry = (
                    f"\nSample {data_idx}:\n"
                    f"  Caption: {metadata['single_caption']}\n"
                    f"  Video Path: {video_path}\n"
                    f"  Ref Images: {metadata['ref_num']}\n"
                    f"  Generated: {gen_video_path}\n"
                    f"  Original: {orig_video_path}\n"
                )
                f_log.write(log_entry)
                print(f"[GPU {gpu_id}] Sample {data_idx} completed")

            except Exception as e:
                error_msg = f"[GPU {gpu_id}] Sample {data_idx} failed: {str(e)}\n"
                f_log.write(error_msg)
                print(error_msg)
                import traceback
                traceback.print_exc()

    print(f"[GPU {gpu_id}] All tasks completed!")


def _extract_video_segment(video_path: str, metadata: dict, output_path: str):
    """
    从原始视频中提取指定片段

    Args:
        video_path: 原始视频路径
        metadata: 数据集元数据
        output_path: 输出路径
    """
    try:
        import imageio

        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        fps = meta.get("fps", 25.0)
        duration = 5.0  # 5秒
        target_frames = 81

        # 获取视频帧数
        try:
            total_frames = reader.count_frames()
        except:
            total_frames = int(meta.get("duration", 5) * fps)

        # 计算采样范围（与数据集保持一致）
        min_index = metadata.get('video_frame_indices', [0])[0]
        max_index = metadata.get('video_frame_indices', [80])[-1]
        frame_indices = metadata.get('video_frame_indices', list(range(81)))

        # 收集帧
        frames = []
        for frame_idx in frame_indices[:target_frames]:
            try:
                frame_data = reader.get_data(int(frame_idx))
                frames.append(frame_data)
            except:
                break

        reader.close()

        # 保存视频
        if frames:
            writer = imageio.get_writer(output_path, fps=15)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
    except Exception as e:
        print(f"Warning: Failed to extract original video: {e}")


def test_on_selected10(args, gpu_id: int, total_gpus: int):
    """
    针对 examples/selected10 数据集的测试函数
    支持多GPU并行：通过 gpu_id 和 total_gpus 分配任务

    Args:
        args: 命令行参数
        gpu_id: 当前GPU编号
        total_gpus: 总GPU数量
    """
    torch.cuda.set_device(gpu_id)

    checkpoint_path = get_checkpoint_path(args)
    output_path = os.path.join("./output", args.visual_log_project_name, "selected10")
    ref_dir, gen_dir, _ = setup_output_dirs(output_path)

    print(f"[GPU {gpu_id}] Loading checkpoint: {checkpoint_path}")

    pipe = load_pipeline(args, checkpoint_path)

    # selected10 数据集路径（可由命令行覆盖）
    test_root = getattr(args, "selected10_path", None) or \
        "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/examples/selected10"

    # 获取所有人物文件夹
    all_folders = sorted([
        f for f in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, f)) and not f.startswith('.')
    ])

    print(f"[GPU {gpu_id}] Found {len(all_folders)} folders: {all_folders}")

    # 根据 gpu_id 分配任务
    my_folders = [f for i, f in enumerate(all_folders) if i % total_gpus == gpu_id]
    print(f"[GPU {gpu_id}] Assigned folders: {my_folders}")

    # 负面提示词
    negative_prompt = ["色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"]

    for folder in my_folders:
        folder_path = os.path.join(test_root, folder)

        # 读取 prompt
        prompt_file = None
        for pf in ["prompt.txt", "prompts.txt"]:
            if os.path.exists(os.path.join(folder_path, pf)):
                prompt_file = os.path.join(folder_path, pf)
                break

        if prompt_file is None:
            print(f"[GPU {gpu_id}] Warning: No prompt file found in {folder}, skipping...")
            continue

        with open(prompt_file, encoding="utf-8") as f:
            text = f.read().strip()

        # 从"三个视角"子文件夹收集图片
        three_view_path = os.path.join(folder_path, "三个视角")
        if not os.path.exists(three_view_path):
            print(f"[GPU {gpu_id}] Warning: '三个视角' folder not found in {folder}, skipping...")
            continue

        image_paths = [
            os.path.join(three_view_path, name)
            for name in os.listdir(three_view_path)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        image_paths.sort()

        if len(image_paths) == 0:
            print(f"[GPU {gpu_id}] Warning: No images found in {three_view_path}, skipping...")
            continue

        print(f"[GPU {gpu_id}] Processing {folder}: {len(image_paths)} images")

        # 打开并处理图片
        ref_images = [Image.open(p).convert("RGB") for p in image_paths]
        ref_images = process_ref_images(ref_images, args.height, args.width)

        # 生成视频
        video, _ = pipe(
            args=args,
            prompt=[text],
            ref_images=[ref_images],
            negative_prompt=negative_prompt,
            seed=42,
            tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            cfg_scale_face=5.0,
            num_ref_images=len(ref_images)
        )

        # 保存视频
        output_video_path = f"{gen_dir}/{folder}.mp4"
        save_video(video, output_video_path, fps=15, quality=10)

        # 保存参考图片
        for i, img in enumerate(ref_images):
            img.save(os.path.join(ref_dir, f"{folder}_ref{i}.png"))

        print(f"[GPU {gpu_id}] Saved: {output_video_path}")

    print(f"[GPU {gpu_id}] All tasks completed!")


def test_on_bench(args, gpu_id: int, total_gpus: int):
    """
    针对 examples/roles 数据集的测试函数
    支持多GPU并行：通过 gpu_id 和 total_gpus 分配任务

    Args:
        args: 命令行参数
        gpu_id: 当前GPU编号
        total_gpus: 总GPU数量
    """
    torch.cuda.set_device(gpu_id)

    checkpoint_path = get_checkpoint_path(args)
    output_path = os.path.join("./output", args.visual_log_project_name, "bench")
    ref_dir, gen_dir, _ = setup_output_dirs(output_path)

    print(f"[GPU {gpu_id}] Loading checkpoint: {checkpoint_path}")

    pipe = load_pipeline(args, checkpoint_path)

    # bench 数据集路径（可由命令行覆盖）
    test_root = getattr(args, "selected10_path", None) or \
        "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/examples/roles"

    # 获取所有人物文件夹
    all_folders = sorted([
        f for f in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, f)) and not f.startswith('.')
    ])

    print(f"[GPU {gpu_id}] Found {len(all_folders)} folders: {all_folders}")

    # 根据 gpu_id 分配任务
    my_folders = [f for i, f in enumerate(all_folders) if i % total_gpus == gpu_id]
    print(f"[GPU {gpu_id}] Assigned folders: {my_folders}")

    # 负面提示词
    negative_prompt = ["色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"]

    for folder in my_folders:
        folder_path = os.path.join(test_root, folder)

        # 读取 prompt
        prompt_file = None
        for pf in ["prompt.txt", "prompts.txt"]:
            if os.path.exists(os.path.join(folder_path, pf)):
                prompt_file = os.path.join(folder_path, pf)
                break

        if prompt_file is None:
            print(f"[GPU {gpu_id}] Warning: No prompt file found in {folder}, skipping...")
            continue

        with open(prompt_file, encoding="utf-8") as f:
            text = f.read().strip()

        # 从"三个视角"子文件夹收集图片
        three_view_path = os.path.join(folder_path, "三个视角")
        if not os.path.exists(three_view_path):
            print(f"[GPU {gpu_id}] Warning: '三个视角' folder not found in {folder}, skipping...")
            continue

        image_paths = [
            os.path.join(three_view_path, name)
            for name in os.listdir(three_view_path)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        image_paths.sort()

        if len(image_paths) == 0:
            print(f"[GPU {gpu_id}] Warning: No images found in {three_view_path}, skipping...")
            continue

        print(f"[GPU {gpu_id}] Processing {folder}: {len(image_paths)} images")

        # 打开并处理图片
        ref_images = [Image.open(p).convert("RGB") for p in image_paths]
        ref_images = process_ref_images(ref_images, args.height, args.width)

        # 生成视频
        video, _ = pipe(
            args=args,
            prompt=[text],
            ref_images=[ref_images],
            negative_prompt=negative_prompt,
            seed=42,
            tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            cfg_scale_face=5.0,
            num_ref_images=len(ref_images)
        )

        # 保存视频
        output_video_path = f"{gen_dir}/{folder}.mp4"
        save_video(video, output_video_path, fps=15, quality=10)

        # 保存参考图片
        for i, img in enumerate(ref_images):
            img.save(os.path.join(ref_dir, f"{folder}_ref{i}.png"))

        print(f"[GPU {gpu_id}] Saved: {output_video_path}")

    print(f"[GPU {gpu_id}] All tasks completed!")


def test_on_original(args):
    """
    使用 datasets/test_set 中 case1-10 数据集的测试函数

    Args:
        args: 命令行参数
    """
    checkpoint_path = get_checkpoint_path(args)
    output_path = os.path.join("./output", args.visual_log_project_name, "original")
    ref_dir, gen_dir, _ = setup_output_dirs(output_path)

    print(f"Loading checkpoint: {checkpoint_path}")

    pipe = load_pipeline(args, checkpoint_path)

    test_root = "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/datasets/test_set"

    # 负面提示词
    negative_prompt = ["色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"]

    for folder in ["case1", "case2", "case3", "case4", "case5", "case6", "case7", "case8", "case9", "case10"]:
        folder_path = os.path.join(test_root, folder)

        # 读取 prompt
        prompt_path = os.path.join(folder_path, "prompt")
        if not os.path.exists(prompt_path):
            print(f"Warning: No prompt file found in {folder}, skipping...")
            continue

        with open(prompt_path, encoding="utf-8") as f:
            text = f.read().strip()

        # 收集图片路径
        image_paths = [
            os.path.join(folder_path, name)
            for name in os.listdir(folder_path)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]
        image_paths.sort()

        if len(image_paths) == 0:
            print(f"Warning: No images found in {folder}, skipping...")
            continue

        # 打开图片
        ref_images = [Image.open(p).convert("RGB") for p in image_paths]
        ref_images = process_ref_images(ref_images, args.height, args.width)

        # 生成视频
        video, _ = pipe(
            args=args,
            prompt=[text],
            ref_images=[ref_images],
            negative_prompt=negative_prompt,
            seed=42,
            tiled=True,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            cfg_scale_face=5.0,
            num_ref_images=len(ref_images)
        )

        # 保存视频
        save_video(video, f"{gen_dir}/{folder}.mp4", fps=15, quality=10)
        print(f"Saved: {folder}.mp4")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    parser = wan_parser()

    # 新增命令行参数
    parser.add_argument('--test_mode', type=str, default='selected10',
                        choices=['train_set', 'selected10', 'original', 'bench'],
                        help='测试模式: train_set (训练集), selected10 (examples/selected10), bench (examples/roles), original (case1-10)')
    parser.add_argument('--num_test_samples', type=int, default=20,
                        help='train_set模式下测试的样本数量')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='可选，覆盖配置文件中的checkpoint路径')
    parser.add_argument('--selected10_path', type=str, default=None,
                        help='selected10数据集路径')

    args, unknown = parser.parse_known_args()
    print("Unknown arguments:", unknown)

    # 解析train.yaml配置
    with open(args.train_yaml, "r", encoding="utf-8") as f:
        conf_info = yaml.safe_load(f)

    # 设置配置参数
    args.dataset_base_path = conf_info["dataset_args"]["base_path"]
    args.max_checkpoints_to_keep = conf_info["train_args"]["max_checkpoints_to_keep"]
    args.resume_from_checkpoint = conf_info["train_args"]["resume_from_checkpoint"]
    args.visual_log_project_name = conf_info["train_args"]["visual_log_project_name"]
    args.seed = conf_info["train_args"]["seed"]
    args.output_path = conf_info["train_args"]["output_path"]
    args.save_steps = conf_info["train_args"]["save_steps"]
    args.save_epoches = conf_info["train_args"]["save_epoches"]
    args.batch_size = conf_info["train_args"]["batch_size"]
    args.local_model_path = conf_info["train_args"]["local_model_path"]
    args.height = conf_info["dataset_args"]["height"]
    args.width = conf_info["dataset_args"]["width"]
    args.num_frames = conf_info["dataset_args"]["num_frames"]
    args.ref_num = conf_info["dataset_args"]["ref_num"]
    args.infer_step = conf_info["infer_args"]["infer_step"]
    args.epoch_id = conf_info["infer_args"]["epoch_id"]
    args.split_rope = conf_info["train_args"]["split_rope"]
    args.split1 = conf_info["train_args"]["split1"]
    args.split2 = conf_info["train_args"]["split2"]
    args.split3 = conf_info["train_args"]["split3"]
    args.split4 = conf_info["train_args"]["split4"]
    args.split5 = conf_info["train_args"]["split5"]

    # 打印配置
    print("=" * 50)
    print("Test Configuration:")
    print(f"  Test Mode: {args.test_mode}")
    print(f"  Checkpoint: {get_checkpoint_path(args)}")
    print(f"  Output Path: {args.output_path}")
    print(f"  Resolution: {args.height}x{args.width}")
    print(f"  Num Frames: {args.num_frames}")
    print(f"  Ref Num: {args.ref_num}")
    print("=" * 50)

    # 根据测试模式选择运行函数
    if args.test_mode == 'train_set':
        # 训练集测试模式 - 支持多GPU并行
        gpu_id = int(os.environ.get("LOCAL_RANK", 0))
        total_gpus = int(os.environ.get("WORLD_SIZE", 1))
        print(f"Running train_set mode with GPU {gpu_id}/{total_gpus}")
        test_on_train_set(args, gpu_id, total_gpus)
    elif args.test_mode == 'selected10':
        # selected10测试模式 - 支持多GPU并行
        gpu_id = int(os.environ.get("LOCAL_RANK", 0))
        total_gpus = int(os.environ.get("WORLD_SIZE", 1))
        print(f"Running selected10 mode with GPU {gpu_id}/{total_gpus}")
        test_on_selected10(args, gpu_id, total_gpus)
    elif args.test_mode == "bench":
        # bench测试模式 - 支持多GPU并行
        gpu_id = int(os.environ.get("LOCAL_RANK", 0))
        total_gpus = int(os.environ.get("WORLD_SIZE", 1))
        print(f"Running bench mode with GPU {gpu_id}/{total_gpus}")
        test_on_bench(args, gpu_id, total_gpus)
    else:
        # original测试模式 - 单GPU
        print("Running original mode")
        test_on_original(args)
