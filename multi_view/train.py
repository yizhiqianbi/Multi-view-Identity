import torch, os, json
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, launch_training_task, wan_parser
from diffsynth.models.wan_video_vae import WanVideoVAE
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import yaml
import torch
import imageio, os, torch, warnings, torchvision, argparse, json
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import matplotlib.pyplot as plt
import os
import re
from datasets.videodataset import MulltiShot_MultiView_Dataset
# from datasets.videodataset_movie import MulltiShot_MultiView_Dataset
# from modules.wanx_module import WanTrainingModule


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        local_model_path=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(local_model_path = local_model_path, model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        

        tokenizer_dir = "/root/paddlejob/workspace/qizipeng/Wan-AI/Wan2.1-T2V-14B/google/umt5-xxl"
        tokenizer_config = ModelConfig(path=tokenizer_dir)

        tokenizer_path = tokenizer_config.path

        # ✅ 兜底：如果 tokenizer_path 是 list，就取第一个并转成所在目录
        if isinstance(tokenizer_path, (list, tuple)):
            tokenizer_path = tokenizer_path[0]
        if tokenizer_path is not None and os.path.isfile(tokenizer_path):
            tokenizer_path = os.path.dirname(tokenizer_path)

        if tokenizer_path is None or not os.path.exists(tokenizer_path):
            tokenizer_path = os.path.join(
                tokenizer_config.local_model_path or "./models",
                tokenizer_config.model_id or "",
                "google/umt5-xxl"
            )

        # tokenizer_path = 
        # print("........................................................")
        # print(tokenizer_config)
        # print("????????????????????!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(tokenizer_path)
        # print("????????????????????!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # pipe.prompter.fetch_tokenizer(tokenizer_path)

        # ✅ Create pipe and get standalone VAE to avoid DeepSpeed ZeRO3 interference
        self.pipe, self.vae = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            redirect_common_files=False
        )

        # self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, redirect_common_files=False)
        
        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)
            
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        # print(use_gradient_checkpointing_offload)
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": [d["single_caption"] for d in data]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # ✅ Add standalone VAE to avoid DeepSpeed ZeRO3 interference
            "vae": self.vae,
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": [d["video"] for d in data],
            "height":  data[0]["video"][0].size[1],
            "width":  data[0]["video"][0].size[0],
            "num_frames": len(data[0]["video"]),
            "ref_images": [d["ref_images"] for d in data],
            "ref_frame_indices": [d.get("ref_frame_indices") for d in data],
            "video_frame_indices": [d.get("video_frame_indices") for d in data],
            "ref_soft_mask_strength": getattr(self, "ref_soft_mask_strength", None),
            "dynamic_ref_weights": getattr(self, "dynamic_ref_weights", [0.1, 0.3, 0.3, 0.3]),  # 动态输入训练
            "ref_mask_debug": getattr(self, "ref_mask_debug", False),
            "ref_mask_debug_path": getattr(self, "ref_mask_debug_path", None),
            "ref_mask_debug_max": getattr(self, "ref_mask_debug_max", None),
            "video_paths": [d.get("video_path") for d in data],
            "ref_poses": [d.get("ref_poses") for d in data],  # Pose-Aware RoPE
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "num_ref_images": data[0]["ref_num"],
            "batch_size": len(data),
            "switch_rope_noise_step": getattr(self, "switch_rope_noise_step", 2000),
        }
        
        # Extra inputs
        # for extra_input in self.extra_inputs:
        #     if extra_input == "input_image":
        #         inputs_shared["input_image"] = data["video"][0]
        #     elif extra_input == "end_image":
        #         inputs_shared["end_image"] = data["video"][-1]
        #     elif extra_input == "reference_image" or extra_input == "vace_reference_image":
        #         inputs_shared[extra_input] = data[extra_input][0]
        #     elif extra_input == "input_video":
        #         inputs_shared["input_pre_video"] = [data["prev_video"][i] for i in range(len(data["prev_video"]))]
        #     elif extra_input == "cropped_images":
        #         inputs_shared["ref_images"] = [data["ref_images"][i] for i in range(len(data["cropped_images"]))]
        #     else:
        #         inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, args, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(args = args, **models, **inputs)
        return loss

    # Required by Accelerate for DeepSpeed compatibility when saving models
    def zero_gather_16bit_weights_on_model_save(self):
        return True


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    args, unknown = parser.parse_known_args()
    print("❗ Unknown arguments:", unknown)
    ### 执行过pip install -e . 的话diffsynth 里的东西修改后要重新安装
    # import pdb; pdb.set_trace()
    ###下面是解析train.yaml里的内容
    with open(args.train_yaml, "r", encoding="utf-8") as f:
        conf_info = yaml.safe_load(f)   # 用 safe_load 更安全
    print(conf_info)
    args.dataset_base_path  = conf_info["dataset_args"]["base_path"]
    args.max_checkpoints_to_keep = conf_info["train_args"]["max_checkpoints_to_keep"]
    args.resume_from_checkpoint =  conf_info["train_args"]["resume_from_checkpoint"]
    args.visual_log_project_name = conf_info["train_args"]["visual_log_project_name"]
    args.seed = conf_info["train_args"]["seed"]
    args.output_path = conf_info["train_args"]["output_path"]
    args.save_steps = conf_info["train_args"]["save_steps"]
    args.save_epoches = conf_info["train_args"]["save_epoches"]
    print("outpath:", args.output_path)
    print("visual_log_project_name:",  args.visual_log_project_name)
    # 检查None值防止字符串拼接错误
    if args.output_path is None or args.visual_log_project_name is None:
        raise ValueError(f"output_path或visual_log_project_name为None: output_path={args.output_path}, visual_log_project_name={args.visual_log_project_name}")
    args.output_path = args.output_path + "/" + args.visual_log_project_name
    args.batch_size = conf_info["train_args"]["batch_size"]
    args.local_model_path = conf_info["train_args"]["local_model_path"]
    args.zero_face_ratio = conf_info["train_args"]["zero_face_ratio"]
    args.dynamic_ref_weights = conf_info["train_args"].get("dynamic_ref_weights", [0.1, 0.3, 0.3, 0.3])  # 动态输入训练
    args.switch_rope_noise_step = conf_info["train_args"].get("switch_rope_noise_step", 2000)
    args.split_rope = conf_info["train_args"]["split_rope"]
    args.split1 = conf_info["train_args"]["split1"]
    args.split2 = conf_info["train_args"]["split2"]
    args.split3 = conf_info["train_args"]["split3"]
    args.split4 = conf_info["train_args"]["split4"]
    args.split5 = conf_info["train_args"]["split5"]
    args.split6 = conf_info["train_args"].get("split6", False)  # Pose-Aware RoPE
    args.pose_rope_max_offset = conf_info["train_args"].get("pose_rope_max_offset", 5)  # Max RoPE offset
    args.roll_mode = conf_info["train_args"].get("roll_mode", "simple")  # Roll mode: "simple" or "geometry"
    args.ref_soft_mask_strength = conf_info["train_args"].get("ref_soft_mask_strength", 0.3)
    args.ref_mask_debug = conf_info["train_args"].get("ref_mask_debug", False)
    args.ref_mask_debug_max = conf_info["train_args"].get("ref_mask_debug_max", 5)
    args.ref_mask_debug_path = conf_info["train_args"].get("ref_mask_debug_path", None)
    # 学习率线性缩放：lr_scaled = lr_base * global_batch_size
    # global_batch_size = batch_size_per_gpu * num_gpus_total
    num_gpus = int(os.environ.get("WORLD_SIZE", 8))  # 从环境变量获取总GPU数，默认单机8卡
    global_batch_size = args.batch_size * num_gpus
    if global_batch_size > 1:
        args.learning_rate = min(args.learning_rate * global_batch_size, args.learning_rate * 10)  # 上限32倍
        print(f"[LR Scaling] base_lr={args.learning_rate / global_batch_size:.2e}, "
              f"global_batch_size={global_batch_size} (batch={args.batch_size} x gpus={num_gpus}), "
              f"scaled_lr={args.learning_rate:.2e}")
    args.height = conf_info["dataset_args"]["height"] 
    args.width = conf_info["dataset_args"]["width"]
    args.num_frames = conf_info["dataset_args"]["num_frames"]
    args.ref_num = conf_info["dataset_args"]["ref_num"]
    args.mask_ref_ratio = conf_info["dataset_args"].get("mask_ref_ratio", 0.0)
    # args.visual_log_project_name = conf_info["train_args"]["visual_log_project_name"]+"_{}".formate(args.height)+"_{}".formate(args.width)

    dataset = MulltiShot_MultiView_Dataset(
        dataset_base_path=args.dataset_base_path,
        resolution=(args.height, args.width),
        ref_num=args.ref_num,
        mask_ref_ratio=args.mask_ref_ratio,
        training=True,
        prefer_fallback=False,
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        local_model_path = args.local_model_path
    )
    model.ref_soft_mask_strength = args.ref_soft_mask_strength
    model.dynamic_ref_weights = args.dynamic_ref_weights  # 动态输入训练
    model.ref_mask_debug = args.ref_mask_debug
    model.ref_mask_debug_max = args.ref_mask_debug_max
    model.switch_rope_noise_step = args.switch_rope_noise_step
    if model.ref_mask_debug:
        rank = int(os.environ.get("RANK", "0"))
        debug_path = args.ref_mask_debug_path
        if not debug_path:
            debug_path = f"/root/paddlejob/workspace/qizipeng/refmask_debug_rank{rank}.jsonl"
        elif not debug_path.endswith(".jsonl"):
            debug_path = os.path.join(debug_path, f"refmask_debug_rank{rank}.jsonl")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        model.ref_mask_debug_path = debug_path
    # Warmup from 1e-5 to 1e-4 in 1000 steps
    start_lr = 1e-5
    target_lr = 2e-5
    warmup_steps = 1000
    
    def warmup_lambda(current_step):
        if current_step < warmup_steps:
            # 线性从 start_lr/target_lr (0.1) 增长到 1.0
            return start_lr / target_lr + (1.0 - start_lr / target_lr) * current_step / warmup_steps
        return 1.0
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=target_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    launch_training_task(
        args,
        dataset, model, optimizer, scheduler,
        num_epochs =args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_path = args.output_path,
        save_steps = args.save_steps, # 新增：每多少步保存一次
        save_epoches = args.save_epoches,
        max_checkpoints_to_keep = args.max_checkpoints_to_keep, # 最多只保留 5 个最新的检查点
        resume_from_checkpoint = args.resume_from_checkpoint, # 新增：从何处恢复
        seed = args.seed,
        visual_log_project_name = args.visual_log_project_name,
    )
