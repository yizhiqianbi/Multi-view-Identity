import imageio, os, torch, warnings, torchvision, argparse, json
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
import wandb
import tempfile
import requests
import random
import decord
import cv2 
import numpy as np
import shutil
import imageio.v3 as iio
import imageio_ffmpeg as ffmpeg
import math
import re
from diffsynth.trainers.timer import get_timers
import time
import glob
from safetensors.torch import save_file, load_file
import math
import random
from typing import List, Dict, Iterable, Iterator, Sequence, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler, DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import math
import random
from collections import defaultdict, deque
from typing import Iterable, List, Sequence, Optional, Callable, Dict

import torch
from torch.utils.data import DataLoader, Sampler, BatchSampler, RandomSampler, DistributedSampler
from accelerate.utils import DataLoaderConfiguration, DeepSpeedPlugin
import random
import matplotlib.pyplot as plt

class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model
    
    
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class ModelLogger:
    """
    一个集成了 Accelerate、W&B、TensorBoard 和文件日志的日志记录器。
    现在它也负责跟踪全局训练步数，并能被 Accelerator 保存和恢复。
    """
    def __init__(self, accelerator: Accelerator, output_path: str):
        self.accelerator = accelerator
        self.output_path = output_path
        
        # 关键状态：全局步数。将被 Accelerator 自动管理。
        self.global_step = 0
        
        # 注册自身，以便 `accelerator.save_state` 和 `load_state` 可以管理其状态。
        self.accelerator.register_for_checkpointing(self)

        self.timers = get_timers()
        
        # --- 日志文件设置 ---
        log_dir = os.path.join(self.output_path, "log")
        if self.accelerator.is_local_main_process:
            os.makedirs(log_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        
        log_file_path = os.path.join(log_dir, f"rank_{self.accelerator.process_index}.log")
        # 'a+' 模式允许读写，如果文件不存在则创建
        self.file_logger = open(log_file_path, "a+") 
        self.log_to_file(f"Logger initialized. Current global_step: {self.global_step}")
    
    def state_dict(self):
        """
        返回需要被保存的状态。这是 `register_for_checkpointing` 要求的方法。
        """
        return {"global_step": self.global_step}

    def load_state_dict(self, state_dict):
        """
        从 state_dict 中加载状态。这是 `register_for_checkpointing` 要求的方法。
        """
        self.global_step = state_dict["global_step"]
        # 在加载状态后记录日志，便于调试
        self.log_to_file(f"Logger state restored. Resumed global_step: {self.global_step}")

    def log_to_file(self, message: str):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.file_logger.write(f"[{timestamp}] {message}\n")
        self.file_logger.flush()

    def on_step_end(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """在每个实际的优化器步骤后调用。"""
        step_time_ms = self.timers.step_time.stop()
        step_time_s = step_time_ms / 1000.0 # 转换为秒

        gathered_losses = self.accelerator.gather_for_metrics(loss.detach())
        avg_loss = torch.mean(gathered_losses).item()
        local_loss = loss.item()
        learning_rate = optimizer.param_groups[0]['lr']

        if self.accelerator.is_local_main_process:
            log_dict = {
                "train/loss": avg_loss,
                "train/learning_rate": learning_rate,
                "perf/step_time_seconds": step_time_s,
                "progress/epoch": self.current_epoch, # 假设 epoch 信息已设置
                "progress/step": self.global_step,
            }
            self.accelerator.log(log_dict, step=self.global_step)

        log_message = (
            f"Epoch: {self.current_epoch} | "
            f"Iter: {self.global_step:06d} | "
            f"Step Time: {step_time_s:.3f}s | "
            f"Local Loss: {local_loss:.4f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"LR: {learning_rate:.6f}"
        )
        self.log_to_file(log_message)
        
        self.global_step += 1
        self.step_start_time = time.time()

    def set_epoch(self, epoch: int):
        """由外部训练循环设置当前 epoch。"""
        self.current_epoch = epoch

    def close(self):
        self.log_to_file("Logger closing.")
        self.file_logger.close()


class CheckpointManager:
    """
    使用 Accelerator 管理检查点的保存和加载。
    - 只保存可训练的权重以节省空间。
    - 自动轮换检查点，只保留最新的 N 个。
    """
    def __init__(self, accelerator: Accelerator, output_path: str, save_steps: int, save_epoches: int, max_to_keep: int, state_dict_converter=lambda x:x):
        self.accelerator = accelerator
        self.checkpoints_dir = os.path.join(output_path)
        self.save_steps = save_steps
        self.save_epoches = save_epoches
        self.max_to_keep = max_to_keep
        self.state_dict_converter = state_dict_converter
        
        if self.accelerator.is_local_main_process:
            os.makedirs(self.checkpoints_dir, exist_ok=True)

    # ------ 获取 DeepSpeed 引擎 ------
    def get_deepspeed_engine(self, model):
        try:
            import deepspeed
            if isinstance(model, deepspeed.DeepSpeedEngine):
                return model
        except Exception:
            pass
        st = getattr(self.accelerator, "state", None)
        if st is not None:
            if getattr(st, "deepspeed_engine", None) is not None:
                return st.deepspeed_engine
            plugin = getattr(st, "deepspeed_plugin", None)
            if plugin is not None:
                if getattr(plugin, "engine", None) is not None:
                    return plugin.engine
                if getattr(plugin, "deepspeed_engine", None) is not None:
                    return plugin.deepspeed_engine
        return None

    def save_checkpoint(self, model: torch.nn.Module, global_step: int, epoch_id: int):
        """
        Zero-2 + Accelerate 场景下的推荐写法：
        - 所有 rank：accelerator.save_state() 保存 deepspeed + optimizer + scheduler + RNG
        - rank0 再额外导出 trainable 权重为 weights.safetensors
        """
        if not (global_step > 0 and global_step % self.save_steps == 0):
            return

        checkpoint_name = f"checkpoint-step-{global_step}-epoch-{epoch_id + 1}"
        save_path = os.path.join(self.checkpoints_dir, checkpoint_name)

        # 1) rank0 创建目录
        if self.accelerator.is_local_main_process:
            os.makedirs(save_path, exist_ok=True)
        self.accelerator.wait_for_everyone()

        # 2) 让 Accelerate 自己处理 DeepSpeed / 优化器 / scheduler / 随机数
        #    注意：所有 rank 都要调用一次
        self.accelerator.print(f"[CKPT] accelerator.save_state -> {save_path}")
        self.accelerator.save_state(save_path)

        # # 3) rank0 额外保存可训练的权重为 safetensors
        # if self.accelerator.is_local_main_process:
        #     with open(os.path.join(save_path, "trainer_state.json"), "w") as f:
        #         json.dump({"global_step": global_step}, f)
        #     state_dict = self.accelerator.get_state_dict(model)
        #     state_dict = self.accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix="pipe.dit.")
        #     state_dict = self.state_dict_converter(state_dict)

        #     weights_path = os.path.join(save_path, "weights.safetensors")
        #     self.accelerator.save(state_dict, weights_path, safe_serialization=True)
        #     self.accelerator.print(f"[CKPT] trainable weights saved -> {weights_path}")
            # 滚动删除旧 ckpt
        if self.accelerator.is_local_main_process:
            self._rotate_checkpoints()
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, model, resume_from_checkpoint):
        if not resume_from_checkpoint:
            return None

        # 1. 找到 checkpoint 路径
        if resume_from_checkpoint == "latest":
            load_path = self._get_load_path(resume_from_checkpoint)
            if not load_path:
                self.accelerator.print("No checkpoint found to resume from. Starting from scratch.")
                return
        else:
            load_path = resume_from_checkpoint

        if load_path is None:
            self.accelerator.print("[Resume] No checkpoint found")
            return None

        self.accelerator.print(f"[Resume] Loading from: {load_path}")

        # 2. 恢复 deepspeed + optimizer + scheduler + RNG + model states
        try:
            self.accelerator.load_state(load_path)
            self.accelerator.print("[Resume] accelerator.load_state() DONE.")
        except Exception as e:
            self.accelerator.print(f"[Resume] ERROR: {e}")

        # 3. 多卡同步
        self.accelerator.wait_for_everyone()
        # self.accelerator.broadcast_parameters(model)

        self.accelerator.print("[Resume] Model fully restored.")
        return load_path


    def _get_load_path(self, resume_from_checkpoint: str | bool) -> str | None:
        """辅助函数，解析 resume_from_checkpoint 并返回有效的加载路径。"""
        if resume_from_checkpoint is True or str(resume_from_checkpoint).lower() == "latest":
            return self.find_latest_checkpoint()
        
        # 如果提供了具体路径，直接返回
        if os.path.isdir(resume_from_checkpoint):
            return resume_from_checkpoint
        return None

    def find_latest_checkpoint(self) -> str | None:
        """在检查点目录中找到最新的检查点。"""
        all_checkpoints = glob.glob(os.path.join(self.checkpoints_dir, "checkpoint-step-*"))
        if not all_checkpoints:
            return None
        
        try:
            latest_checkpoint = max(
                all_checkpoints, 
                key=lambda path: int(re.search(r'checkpoint-step-(\d+)', path).group(1))
            )
            return latest_checkpoint
        except (ValueError, AttributeError):
            # 如果文件夹命名不规范，无法解析出数字，则返回 None
            return None

    def _rotate_checkpoints(self):
        """删除旧的检查点，只保留 `max_to_keep` 个。"""
        if self.max_to_keep <= 0:
            return

        all_checkpoints = glob.glob(os.path.join(self.checkpoints_dir, "checkpoint-step-*"))
        
        # 解析步数并排序，最旧的在前
        # 使用宽松的正则，只匹配 step 数字，兼容不同的命名格式
        def get_step_number(path):
            match = re.search(r'checkpoint-step-(\d+)', path)
            return int(match.group(1)) if match else 0
        
        try:
            # 过滤掉无法解析的路径
            valid_checkpoints = [p for p in all_checkpoints if re.search(r'checkpoint-step-(\d+)', p)]
            sorted_checkpoints = sorted(valid_checkpoints, key=get_step_number)
        except (ValueError, AttributeError) as e:
            self.accelerator.print(f"[Warning] Could not sort checkpoints for rotation: {e}")
            return

        # 如果检查点数量超过限制，则删除最旧的
        num_to_delete = len(sorted_checkpoints) - self.max_to_keep
        if num_to_delete > 0:
            checkpoints_to_delete = sorted_checkpoints[:num_to_delete]
            for ckpt_path in checkpoints_to_delete:
                self.accelerator.print(f"Deleting old checkpoint: {ckpt_path}")
                shutil.rmtree(ckpt_path)



def launch_training_task(
    args,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    output_path: str = "./models/train",
    save_steps: int = 10, # 新增：每多少步保存一次
    save_epoches: int = 1,
    max_checkpoints_to_keep=5, # 最多只保留 5 个最新的检查点
    resume_from_checkpoint: str | bool = "latest", # 新增：从何处恢复
    seed: int = 42,
    visual_log_project_name: str=None,
):
    def collate_fn_identity(batch):
        return batch 
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size = args.batch_size, collate_fn = collate_fn_identity, num_workers = 8)
    # dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0])
    # dispatch_batches=False 可以绕过accelerator自动的组建batch，组batch 操作放在./dataset/collate_fns/下自定义
    # --- 1. 初始化 Accelerator 和组件 ---
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        # log_with=["wandb"],
        project_dir=output_path,
    )
    set_seed(seed, device_specific=True)
    # if visual_log_project_name is None:
    #     visual_log_project_name = os.path.join(output_path, "wandb")
    # 在主进程上初始化 tracker
    if accelerator.is_local_main_process:
        accelerator.init_trackers(project_name=visual_log_project_name)
    model_logger = ModelLogger(accelerator, output_path)
    checkpoint_manager = CheckpointManager(accelerator, output_path, save_steps, save_epoches, max_checkpoints_to_keep)

    # --- 2. 准备所有组件 ---
    # 注意：顺序很重要，必须先 prepare 再 load_state
    # 因为 DeepSpeed 引擎需要在 prepare 后才能正确加载状态
    pipe, optimizer, dataloader, scheduler = accelerator.prepare(
        model.pipe, optimizer, dataloader, scheduler
    )
    model.pipe = pipe

    # --- 3. 加载检查点状态 ---
    if resume_from_checkpoint:
        load_path = checkpoint_manager.load_checkpoint(model, "latest")
        if load_path is not None:
            trainer_state_path = os.path.join(load_path, "trainer_state.json")
            if os.path.exists(trainer_state_path):
                with open(trainer_state_path, "r") as f:
                    state = json.load(f)
                global_step = state.get("global_step", 0)
                model_logger.global_step = global_step
                accelerator.print(f"[Resume] Restored global_step: {global_step}")
            else:
                accelerator.print(f"[Resume] trainer_state.json not found, starting from step 0")
        else:
            accelerator.print("[Resume] No checkpoint found, starting from scratch")
    
    # --- 4. 学习率处理 ---
    # TODO: 学习率恢复可能不对，目前依赖 scheduler 自动恢复
    # 如需手动设置学习率，可在此处添加逻辑


    # --- 4. 计算恢复后的起始点 ---
    # model_logger.global_step 已经被 load_state 恢复了
    total_steps_per_epoch = len(dataloader)
    resume_step = model_logger.global_step % total_steps_per_epoch
    starting_epoch = model_logger.global_step // total_steps_per_epoch
    
    accelerator.print("--- Starting Training ---")
    accelerator.print(f"Num Epochs: {num_epochs}")
    accelerator.print(f"Total steps per epoch: {total_steps_per_epoch}")
    accelerator.print(f"Resuming from Epoch: {starting_epoch}, Step: {resume_step}")
    accelerator.print(f"Total number of data: {len(dataloader)}")

    timers = get_timers()
    if accelerator.is_local_main_process:
        save_loss_path = f"/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/training_loss_plots/{args.visual_log_project_name}"
        loss_means = []
        os.makedirs(save_loss_path, exist_ok=True)
    for epoch_id in range(starting_epoch, num_epochs):
        model.train()
        model_logger.set_epoch(epoch_id)
        
        # 如果是恢复的第一个 epoch，需要跳过已经训练过的 steps
        if epoch_id == starting_epoch and resume_step > 0:
            # 使用 enumerate 和 islice 的组合来跳过
            active_dataloader = enumerate(dataloader)
            for _ in range(resume_step):
                next(active_dataloader)
        else:
            active_dataloader = enumerate(dataloader)

        pbar = tqdm(
            active_dataloader, 
            initial=resume_step if epoch_id == starting_epoch else 0,
            total=total_steps_per_epoch,
            disable=not accelerator.is_local_main_process, 
            desc=f"Epoch {epoch_id}"
        )

        for step, data in pbar:
            timers.step_time.start()
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data, args)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                # 只在真正的优化步记录（累积步结束）
                if accelerator.sync_gradients:
                    model_logger.on_step_end(loss, optimizer)
                    checkpoint_manager.save_checkpoint(model, model_logger.global_step, epoch_id)
                    # 跨卡聚合 loss（每卡一个标量），取全局平均
                    loss_mean = accelerator.gather(loss.detach()).mean().item()
                if model_logger.global_step % 10 == 0:
                    if accelerator.is_local_main_process:
                        loss_means.append(loss_mean)
                        plt.figure(figsize=(8, 5))
                        plt.plot(loss_means, marker="o", linestyle="-", label="Training Loss")
                        plt.xlabel("X (every 10 steps)")
                        plt.ylabel("Loss")
                        plt.title(f"Loss Curve up to step {model_logger.global_step}")
                        plt.grid(True)
                        plt.legend()
                        plt.savefig(f"{save_loss_path}/loss_mean.png")
                        plt.close()

    # 重置 resume_step，因为后续 epoch 都是从头开始   
    resume_step = 0

    # --- 6. 训练结束 ---
    accelerator.wait_for_everyone()
    accelerator.print("--- Training Finished ---")
    
    # 保存最终的模型
    if accelerator.is_local_main_process:
        accelerator.print("Saving final model...")
        final_save_path = os.path.join(checkpoint_manager.checkpoints_dir, "final_model")
        accelerator.save_state(final_save_path)
    
    accelerator.end_training()
    model_logger.close()

    # if accelerator.is_local_main_process:
    #     wandb.finish()


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=False, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./checkpoints", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--save_steps", type=int, default=10, help="The epoch to save.")
    parser.add_argument("--train_yaml", type=str, default="../../../../conf/config.yaml", help="The train yaml file.")
    parser.add_argument("--max_checkpoints_to_keep", type=int, default=5, help="max_checkpoints_to_keep")
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False, help="The dataset yaml file")
    parser.add_argument("--seed", type=int, default=42, help="The random seed")
    parser.add_argument("--visual_log_project_name", type=str, default=None, help="Project name")
    parser.add_argument("--max_frames_per_batch", type=int, default=136, help="Max frames num for each batch")
    parser.add_argument("--prompt_index", type=int, default=-1, help="Prompt index for generation")
    parser.add_argument("--ref_num", type=int, default=3, help="The number of reference images")
    parser.add_argument("--local_model_path", type=str, default="", help="The default root path of the Wan weights")
    parser.add_argument("--batch_size", type=int, default=1, help="The default batch size")
    parser.add_argument("--save_epoches", type=int, default=1, help="The default saving epoch")
    parser.add_argument("--split_rope", type=bool, default=False, help="Whether apply different rope into reference images ")
    parser.add_argument("--split1", type=bool, default=False, help=" ")
    parser.add_argument("--split2", type=bool, default=False, help=" ")
    parser.add_argument("--split3", type=bool, default=False, help=" ")
    parser.add_argument("--split4", type=bool, default=False, help=" ")
    parser.add_argument("--split5", type=bool, default=False, help=" ")

    return parser


if __name__ == '__main__':

    dataset = VideoDataset(use_history = True, base_path = "/user/kg-aigc/rd_dev/qizipeng/luo_data_sorted_full.json")

    train_dataset = dataset.data
    for d in tqdm(train_dataset[7772:7773]):
        # video = dataset.get_video_from_path(d["video_path"], d["video_time"], is_history=False)
        # if len(video) != 81 :
        #     print(d)

        video =  dataset.get_video_from_path(
                            d["videoHis_path"], d["videoHis_time"], is_history=True
                        )
        if video == None:
            print(d)


    import pdb; pdb.set_trace()



