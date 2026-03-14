import imageio, os, torch, warnings, torchvision, argparse, json
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import random
from decord import VideoReader
from decord import cpu, gpu
import imageio.v3 as iio

from torchvision import transforms
import torchvision
import random
import decord
from torchvision import transforms
import re
decord.bridge.set_bridge('torch')
import random
import numpy as np
from PIL import Image, ImageOps

class MulltiShot_MultiView_Dataset(torch.utils.data.Dataset):
    # 预计算文件路径（类变量）
    PRECOMPUTED_JSON_PATH = '/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/conf/filterd_useable.json'
    
    def __init__(self, dataset_base_path='/user/kg-aigc/rd_dev/jiahaochen/all_videos_wangpan_duration150.data',
                        # ref_image_path='/root/paddlejob/workspace/qizipeng/code/longvideogen/output.json',  # DEPRECATED: 不再使用
                        condition_base_path='/user/kg-aigc/rd_dev/jiahaochen/grounded_sam_wangpan_hq',
                        condition_fallback_path='/user/kg-aigc/rd_dev/jiahaochen/face_duration150',
                        time_division_factor=4, 
                        time_division_remainder=1,
                        max_pixels=1920*1080,
                        height_division_factor=16, width_division_factor=16,
                        transform=None,
                        length=None,
                        resolution=None,
                        prev_length=5,
                        ref_num = 3,
                        mask_ref_ratio=0.0,
                        mask_ref_patch_size=None,
                        training = True,
                        precomputed_json_path = None,  # 预计算文件路径
                        prefer_fallback = False):  # 新增参数：是否优先使用 fallback 人脸数据
        self.data_path = dataset_base_path
        self.condition_base_path = condition_base_path
        self.condition_fallback_path = condition_fallback_path
        self.data = []
        self.length = length
        self.resolution = resolution  
        self.height, self.width = resolution
        self.num_frames = length
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.prev_length = prev_length
        self.training = training
        self.ref_num = ref_num
        self.prefer_fallback = prefer_fallback  # 是否优先使用 fallback 人脸数据
        self.mask_ref_ratio = 0.0 if mask_ref_ratio is None else float(mask_ref_ratio)
        if self.mask_ref_ratio < 0.0:
            self.mask_ref_ratio = 0.0
        if self.mask_ref_ratio > 1.0:
            self.mask_ref_ratio = 1.0
        # Pixel-space patch size for MAE-style masking (default aligns with 16x16 patches).
        self.mask_ref_patch_size = self.height_division_factor if mask_ref_patch_size is None else int(mask_ref_patch_size)
        if self.mask_ref_patch_size <= 0:
            self.mask_ref_patch_size = self.height_division_factor
        
        if self.prefer_fallback:
            print(f"🔄 人脸数据优先级: fallback > primary")
        else:
            print(f"🔄 人脸数据优先级: primary > fallback")
        
        # 确定预计算文件路径
        if precomputed_json_path is not None:
            self.precomputed_json_path = precomputed_json_path
        else:
            self.precomputed_json_path = self.PRECOMPUTED_JSON_PATH

        # ============================================================f
        # [NEW] 优先加载预计算的 JSON 文件，跳过耗时验证
        # ============================================================
        if os.path.exists(self.precomputed_json_path):
            print(f"✅ 发现预计算文件，直接加载: {self.precomputed_json_path}")
            with open(self.precomputed_json_path, 'r', encoding='utf-8') as f:
                meta_datas_list = json.load(f)
            print(f"✅ 从预计算文件加载了 {len(meta_datas_list)} 条数据")
            
            # 快速构建 self.data（无需验证，预计算已验证过）
            for context in tqdm(meta_datas_list, desc="加载预计算数据"):
                disk_path = context.get("disk_path", "")
                # 规范化路径，处理双斜杠 // 等问题
                disk_path = os.path.normpath(disk_path)
                if not disk_path.lower().endswith(".mp4"):
                    continue
                
                video_filename = os.path.basename(disk_path)
                condition_path_primary = os.path.join(self.condition_base_path, video_filename)
                condition_path_fallback = os.path.join(self.condition_fallback_path, video_filename)
                condition_path = condition_path_primary if os.path.isdir(condition_path_primary) else condition_path_fallback
                
                min_index = context.get("min_index", 0)
                max_index = context.get("max_index", 0)
                fps = context.get("raw_meta", {}).get("fps", 25.0)
                
                meta_prompt = {}
                meta_prompt["global_caption"] = None
                meta_prompt["per_shot_prompt"] = []
                meta_prompt["single_prompt"] = context.get('text', '')

                self.data.append({
                    'video_path': disk_path,
                    'meta_prompt': meta_prompt,
                    'condition_path': condition_path,
                    'min_index': min_index,
                    'max_index': max_index,
                    'fps': fps,
                    'facedetect_v1': context.get('facedetect_v1', []),  # [NEW] 保存 pose 数据
                    'facedetect_v1_frame_index': context.get('facedetect_v1_frame_index', [])  # [NEW] 保存帧索引
                })
        else:
            # ============================================================
            # 预计算文件不存在，使用原有验证逻辑
            # ============================================================
            print(f"⚠️ 预计算文件不存在: {self.precomputed_json_path}")
            print(f"📖 将从原始数据加载并验证（较慢）...")
            
            # 支持两种格式：JSON Lines（每行一个JSON）和标准 JSON 字典
            meta_datas_list = []
            with open(self.data_path, 'r', encoding='utf-8') as f:
                first_char = f.read(1)
                f.seek(0)  # 重置到文件开头
                
                if first_char == '{':
                    # JSON Lines 格式：每行一个 JSON 对象
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                meta_datas_list.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                elif first_char == '[':
                    # 标准 JSON 数组格式
                    meta_datas_list = json.load(f)
                else:
                    # 尝试作为 JSON 字典格式（旧格式）
                    f.seek(0)
                    meta_datas_dict = json.load(f)
                    meta_datas_list = list(meta_datas_dict.values())
        
            for context in tqdm(meta_datas_list):
                disk_path = context.get("disk_path", "")
                # 规范化路径，处理双斜杠 // 等问题
                disk_path = os.path.normpath(disk_path)
                if not disk_path.lower().endswith(".mp4"):
                    continue

                # ============================================================
                # [NEW] 从预计算目录读取人脸图片，不再进行实时人脸角度计算
                # ============================================================
                video_filename = os.path.basename(disk_path)
                condition_path_primary = os.path.join(self.condition_base_path, video_filename)
                condition_path_fallback = os.path.join(self.condition_fallback_path, video_filename)
                
                # 检查预计算人脸目录是否存在（至少一个目录存在）
                if not os.path.isdir(condition_path_primary) and not os.path.isdir(condition_path_fallback):
                    # 两个目录都不存在，跳过该视频
                    continue
                
                # 合并两个目录的人脸图片文件名（取并集）
                primary_files = set()
                fallback_files = set()
                if os.path.isdir(condition_path_primary):
                    primary_files = set(f for f in os.listdir(condition_path_primary) if f.endswith('_crop_face.png'))
                if os.path.isdir(condition_path_fallback):
                    fallback_files = set(f for f in os.listdir(condition_path_fallback) if f.endswith('_crop_face.png'))
                
                all_face_files = sorted(primary_files | fallback_files)
                
                # 验证图片是否能成功打开，统计有效图片数量
                valid_face_count = 0
                for face_file in all_face_files:
                    primary_path = os.path.join(condition_path_primary, face_file)
                    fallback_path = os.path.join(condition_path_fallback, face_file)
                    
                    # 优先尝试 primary，失败则尝试 fallback
                    img_valid = False
                    if face_file in primary_files:
                        try:
                            img = Image.open(primary_path)
                            img.verify()  # 验证图片完整性
                            img_valid = True
                        except Exception:
                            pass
                    
                    if not img_valid and face_file in fallback_files:
                        try:
                            img = Image.open(fallback_path)
                            img.verify()
                            img_valid = True
                        except Exception:
                            pass
                    
                    if img_valid:
                        valid_face_count += 1
                        # 已经够3张了，不需要继续验证
                        if valid_face_count >= self.ref_num:
                            break
                
                # 有效图片不足3张，跳过该视频
                if valid_face_count < self.ref_num:
                    continue
                
                # 保存主目录路径（用于后续加载时的优先级判断）
                condition_path = condition_path_primary if os.path.isdir(condition_path_primary) else condition_path_fallback
                
                # 读取 min_index, max_index, fps
                min_index = context.get("min_index", 0)
                max_index = context.get("max_index", 0)
                fps = context.get("raw_meta", {}).get("fps", 25.0)
                
                meta_prompt = {}
                meta_prompt["global_caption"] = None
                meta_prompt["per_shot_prompt"] = []
                meta_prompt["single_prompt"] = context.get('text', '')

                self.data.append({
                    'video_path': disk_path,
                    'meta_prompt': meta_prompt,
                    'condition_path': condition_path,
                    'min_index': min_index,
                    'max_index': max_index,
                    'fps': fps,
                    'facedetect_v1': context.get('facedetect_v1', []),  # [NEW] 保存 pose 数据
                    'facedetect_v1_frame_index': context.get('facedetect_v1_frame_index', [])  # [NEW] 保存帧索引
                })

            # ============================================================
            # [DEPRECATED] 原有人脸角度筛选逻辑 - 已注释
            # ============================================================
            # # reader = imageio.get_reader(meta_datas[video_path]["disk_path"])
            # # total_original_frames = reader.count_frames()
            # # total_frame = total_original_frames # context["end_index"] - context["start_index"] - 1
            # total_frame = None
            # ref_id  = self.get_ref_id(face_crop_angle = context['facedetect_v1'], facedetect_v1_frame_index = context['facedetect_v1_frame_index'], total_frame = total_frame)
            # if ref_id == []:
            #     continue
            # ref_id_all = []
            # for ids in ref_id:
            #     ref_id_grop = []
            #     for id in ids:
            #         coordinate = context['facedetect_v1'][id][0]['detect']
            #         if context['facedetect_v1'][id][0]['detect']["prob"] < 0.99:
            #             continue
            #         top, height, width, left = coordinate['top'], coordinate['height'], coordinate['width'], coordinate['left']
            #         if not(min(height, width) > 80 ):
            #             continue
            #         # enlarge bbox 1.5x
            #         width = int(width * 1)
            #         height = int(height * 1)
            #         frame_index = context['facedetect_v1_frame_index'][id]
            #         ref_id_grop.append([top, height, width, left, int(frame_index)])
            #     if ref_id_grop != []:
            #         if len(ref_id_grop) >= 3: #self.ref_num: ### 为了和ref_num = 3 保持数据一致
            #             ref_id_all.append(ref_id_grop)
            # if ref_id_all == []:
            #     continue
            # meta_prompt = {}
            # meta_prompt["global_caption"] = None
            # meta_prompt["per_shot_prompt"] = []
            # meta_prompt["single_prompt"] = context['text']
            # self.data.append({'video_path': disk_path, 'meta_prompt': meta_prompt, "ref_id_all": ref_id_all})
            # # self.data.append({'video_path':video_path, 'meta_prompt': meta_prompt, "ref_id_all": ref_id_all})
            # ============================================================

        random.seed(42)  # 让每次划分一致（可选）
        total = len(self.data)
        test_count = min(200, total)  # [MODIFIED] 固定 200 个测试样本

        # 随机选择 test 的 index
        test_indices = set(random.sample(range(total), test_count))

        self.data_test = [self.data[i] for i in range(total) if i in test_indices]
        self.data_train = [self.data[i] for i in range(total) if i not in test_indices]
        print(f"🔥 数据集划分完成：Train={len(self.data_train)}, Test={len(self.data_test)}")

        if self.height is not None and self.width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif self.height is None and self.width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True

        # ============================================================
        # [NEW] Pose 数据相关方法
        # ============================================================
    def _get_pose_for_frame_index(self, facedetect_v1, facedetect_v1_frame_index, target_frame_index):
        """
        根据帧索引获取对应的 pose 信息（仅精准匹配）

        Args:
            facedetect_v1: 人脸检测数据列表，每个元素包含 angle 信息
            facedetect_v1_frame_index: 人脸检测对应的帧索引列表
            target_frame_index: 目标帧索引

        Returns:
            dict: {'yaw': float, 'pitch': float, 'roll': float}，如果找不到返回 None
        """
        # 仅精确匹配
        for i, frame_idx in enumerate(facedetect_v1_frame_index):
            if frame_idx == target_frame_index:
                # 精确匹配
                if i < len(facedetect_v1) and facedetect_v1[i] and len(facedetect_v1[i]) > 0:
                    angle = facedetect_v1[i][0].get('angle', {})
                    return {
                        'yaw': float(angle.get('yaw', 0.0)),
                        'pitch': float(angle.get('pitch', 0.0)),
                        'roll': float(angle.get('roll', 0.0))
                    }

        # 没有精确匹配，返回 None
        return None

    def _normalize_pose_binned(self, yaw, pitch, roll):
        """
        将 pose 使用分箱（bin）方式归一化到 [0, 1] 区间

        对于周期性角度，分成不同区间进行编码：
        - yaw: (-180, 180] -> 8 个区间，每 45 度一个区间
        - pitch: (-90, 90] -> 4 个区间，每 45 度一个区间
        - roll: (-180, 180] -> 8 个区间，每 45 度一个区间

        超出范围的值先用模运算拉回有效范围，再分箱。

        Args:
            yaw: 偏航角（度），原始范围约 [-100, 98]，可能有超出范围的值
            pitch: 俯仰角（度），范围约 [-30, 57]
            roll: 翻滚角（度），原始范围约 [-187, 181]，可能有超出范围的值

        Returns:
            dict: {'yaw': float, 'pitch': float, 'roll': float}，每个值在 [0, 1] 范围内
        """
        import math

        # 周期性处理：将角度映射到有效范围
        yaw = yaw % 360
        if yaw > 180:
            yaw = yaw - 360
        elif yaw < -180:
            yaw = yaw + 360

        roll = roll % 360
        if roll > 180:
            roll = roll - 360
        elif roll < -180:
            roll = roll + 360

        # pitch: (-90, 90]，非周期性但可能超出范围
        if pitch > 90:
            pitch = 90
        elif pitch < -90:
            pitch = -90

        # 分箱归一化：将角度映射到离散的 bin 值
        # yaw 和 roll: 8 个区间，每 45 度一个区间，归一化到 [0, 1/7, ..., 1]
        # pitch: 4 个区间，每 45 度一个区间，归一化到 [0, 1/3, ..., 1]
        yaw_norm = (yaw + 180) / 360  # [-180, 180] -> [0, 1]
        pitch_norm = (pitch + 90) / 180   # [-90, 90] -> [0, 1]
        roll_norm = (roll + 180) / 360  # [-180, 180] -> [0, 1]

        # 线性插值到 [0, 1] 范围
        yaw_norm = max(0.0, min(1.0, yaw_norm))
        pitch_norm = max(0.0, min(1.0, pitch_norm))
        roll_norm = max(0.0, min(1.0, roll_norm))

        return {
            'yaw': yaw_norm,
            'pitch': pitch_norm,
            'roll': roll_norm
        }

    def _normalize_pose_sin_cos(self, yaw, pitch, roll):
        """
        将 pose 使用 sin/cos 编码归一化到 [0, 1] 区间（考虑周期性）

        对于周期性角度（yaw, roll），使用 sin/cos 编码可以保持周期连续性。
        sin(angle) 和 cos(angle) 范围是 [-1, 1]，映射到 [0, 1]。

        Args:
            yaw: 偏航角（度），原始范围约 [-100, 98]，可能有超出范围的值
            pitch: 俯仰角（度），范围约 [-30, 57]
            roll: 翻滚角（度），原始范围约 [-187, 181]，可能有超出范围的值

        Returns:
            dict: {
                'yaw_sin': float, 'yaw_cos': float,
                'pitch_sin': float, 'pitch_cos': float,
                'roll_sin': float, 'roll_cos': float
            }，每个值在 [0, 1] 范围内
        """
        import math

        # 周期性处理：将角度映射到有效范围
        yaw = yaw % 360
        if yaw > 180:
            yaw = yaw - 360
        elif yaw < -180:
            yaw = yaw + 360

        roll = roll % 360
        if roll > 180:
            roll = roll - 360
        elif roll < -180:
            roll = roll + 360

        # pitch: (-90, 90]，非周期性但可能超出范围
        if pitch > 90:
            pitch = 90
        elif pitch < -90:
            pitch = -90

        # 将角度转换为弧度
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)

        # sin/cos 编码，然后归一化到 [0, 1]
        return {
            'yaw_sin': (math.sin(yaw_rad) + 1) / 2,
            'yaw_cos': (math.cos(yaw_rad) + 1) / 2,
            'pitch_sin': (math.sin(pitch_rad) + 1) / 2,
            'pitch_cos': (math.cos(pitch_rad) + 1) / 2,
            'roll_sin': (math.sin(roll_rad) + 1) / 2,
            'roll_cos': (math.cos(roll_rad) + 1) / 2,
        }

    # ============================================================
    # [DEPRECATED] get_ref_id 方法 - 不再使用，人脸数据改为预计算
    # ============================================================
    # def get_ref_id(self, face_crop_angle, facedetect_v1_frame_index = None, total_frame = None, angle_threshold=50):
    #     """
    #         返回满足角度差异要求的三元组 [i, j, k]
    #         要求：
    #             - face_crop_angle[i] / [j] / [k] 都必须非空
    #             - i,j 两者任意 yaw/pitch/roll 差值 > angle_threshold
    #             - k != i != j，且 k 也必须非空
    #         """
    #     ref_id = []
    #     max_try = 5
    #     need_max = 3
    #     try_num = 0
    #         
    #     # 过滤空元素，保留有效索引
    #     valid_indices = [idx for idx, item in enumerate(face_crop_angle) if item]
    #     N = len(valid_indices)
    #
    #     if N < 3:
    #         return ref_id   # 不足 3 张有效图，无法组成三元组
    #
    #     # 两两组合检查角度差
    #     for a in range(N - 1):
    #         i = valid_indices[a]
    #         # if facedetect_v1_frame_index[i] > total_frame:
    #         #     continue
    #         angle_i = face_crop_angle[i][0]["angle"]
    #
    #         for b in range(a + 1, N):
    #             j = valid_indices[b]
    #             # if facedetect_v1_frame_index[j] > total_frame:
    #             #     continue
    #             angle_j = face_crop_angle[j][0]["angle"]
    #
    #             # 判断是否满足阈值
    #             if (
    #                 abs(angle_i["yaw"]   - angle_j["yaw"])   > angle_threshold or
    #                 abs(angle_i["pitch"] - angle_j["pitch"]) > angle_threshold or
    #                 abs(angle_i["roll"]  - angle_j["roll"])  > angle_threshold
    #             ):
    #                 # 找第三个 k
    #                 for c in range(N):
    #                     k = valid_indices[c]
    #                     # if facedetect_v1_frame_index[k] > total_frame:
    #                     #     continue
    #                     if k != i and k != j:
    #                         ref_id.append([i, j, k])
    #                         break
    #
    #                 try_num += 1
    #                 if try_num >= max_try or len(ref_id) >= need_max:
    #                     return ref_id
    #
    #     return ref_id
    # ============================================================
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    # def 
    #     img_ratio = img.width / img.height
    #     target_ratio = w / h
    #     if img_ratio > target_ratio:  # Image is wider than target
    #         new_width = w
    #         new_height = int(new_width / img_ratio)
    #     else:  # Image is taller than target
    #         new_height = h
    #         new_width = int(new_height * img_ratio)
        
    #     # img = img.resize((new_width, new_height), Image.ANTIALIAS)
    #     img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    #     # Create a new image with the target size and place the resized image in the center
    #     delta_w = w - img.size[0]
    #     delta_h = h - img.size[1]
    #     padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    #     new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))

    def resize_ref(self, img, target_h, target_w):
        h = target_h
        w = target_w
        img = img.convert("RGB")
        # Calculate the required size to keep aspect ratio and fill the rest with padding.
        img_ratio = img.width / img.height
        target_ratio = w / h
        
        if img_ratio > target_ratio:  # Image is wider than target
            new_width = w
            new_height = int(new_width / img_ratio)
        else:  # Image is taller than target
            new_height = h
            new_width = int(new_height * img_ratio)
        
        # img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create a new image with the target size and place the resized image in the center
        delta_w = w - img.size[0]
        delta_h = h - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))

        return new_img

    def apply_mae_mask(self, img):
        if self.mask_ref_ratio <= 0.0:
            return img
        patch = self.mask_ref_patch_size
        if patch is None or patch <= 0:
            return img
        w, h = img.size
        grid_h = h // patch
        grid_w = w // patch
        if grid_h <= 0 or grid_w <= 0:
            return img
        num_patches = grid_h * grid_w
        num_mask = int(round(num_patches * self.mask_ref_ratio))
        if num_mask <= 0:
            return img
        if num_mask >= num_patches:
            arr = np.array(img)
            arr[:grid_h * patch, :grid_w * patch] = 0
            return Image.fromarray(arr)

        mask = np.zeros(num_patches, dtype=np.uint8)
        mask[:num_mask] = 1
        np.random.shuffle(mask)
        mask = mask.reshape(grid_h, grid_w)

        arr = np.array(img)
        h0, w0 = grid_h * patch, grid_w * patch
        mask_up = np.kron(mask, np.ones((patch, patch), dtype=np.uint8)).astype(bool)
        arr_slice = arr[:h0, :w0]
        arr_slice[mask_up] = 0
        arr[:h0, :w0] = arr_slice
        return Image.fromarray(arr)


    def load_video_crop_ref_image(self, video_path=None, condition_path=None, min_index=0, max_index=0, fps=25.0,
                                  facedetect_v1=None, facedetect_v1_frame_index=None):
        """
        加载视频帧和预计算的参考人脸图片

        Args:
            video_path: 视频文件路径
            condition_path: 预计算人脸图片目录路径
            min_index: 必须包含的最小帧索引
            max_index: 必须包含的最大帧索引
            fps: 原视频帧率
            facedetect_v1: 人脸检测数据列表（包含 pose 信息）
            facedetect_v1_frame_index: 人脸检测对应的帧索引列表

        Returns:
            frames: 视频帧列表
            ref_images: 参考人脸图片列表
            ref_frame_indices: 参考图片对应的帧索引列表
            sample_ids: 采样的视频帧索引列表
            ref_poses: 参考图片对应的 pose 信息列表 [{'yaw': x, 'pitch': y, 'roll': z}, ...]
        """
        ### fps 转化
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        original_fps = meta.get("fps", fps)
        target_fps = 16
        duration_seconds = 5
        target_frames = target_fps * duration_seconds + 1   # = 81 frames

        # ---- 获取原视频帧数 ----
        try:
            total_original_frames = reader.count_frames()
        except:
            total_original_frames = int(meta.get("duration", 5) * original_fps)

        # ============================================================
        # [NEW] 采样策略：必须覆盖 [min_index, max_index] 的 5 秒
        # ============================================================
        # 需要多少原始帧（5秒）
        need_orig_frames = int(original_fps * duration_seconds)
        
        # 计算包含 [min_index, max_index] 的采样范围
        # 以 min_index 和 max_index 的中点为中心，向两侧扩展
        center_frame = (min_index + max_index) // 2
        half_frames = need_orig_frames // 2
        
        segment_start = center_frame - half_frames
        segment_end = center_frame + (need_orig_frames - half_frames)
        
        # 边界检查：确保不超出视频范围
        if segment_start < 0:
            segment_start = 0
            segment_end = min(need_orig_frames, total_original_frames)
        elif segment_end > total_original_frames:
            segment_end = total_original_frames
            segment_start = max(0, total_original_frames - need_orig_frames)
        
        # 确保采样范围覆盖 [min_index, max_index]
        if segment_start > min_index:
            segment_start = min_index
            segment_end = min(segment_start + need_orig_frames, total_original_frames)
        if segment_end < max_index:
            segment_end = min(max_index + 1, total_original_frames)
            segment_start = max(0, segment_end - need_orig_frames)

        # ---- 均匀采样 81 帧 ----
        sample_ids = np.linspace(segment_start, segment_end - 1, num=target_frames, dtype=int)

        frames = []
        for frame_id in sample_ids:
            frame = reader.get_data(int(frame_id))
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        
        reader.close()

        # ============================================================
        # [NEW] 从预计算目录读取参考人脸图片（primary + fallback 合并策略）
        # ============================================================
        video_filename = os.path.basename(video_path)
        primary_dir = os.path.join(self.condition_base_path, video_filename)
        fallback_dir = os.path.join(self.condition_fallback_path, video_filename)
        
        # 获取两个目录中所有的人脸文件名（取并集）
        primary_files = set()
        fallback_files = set()
        if os.path.isdir(primary_dir):
            primary_files = set(f for f in os.listdir(primary_dir) if f.endswith('_crop_face.png'))
        if os.path.isdir(fallback_dir):
            fallback_files = set(f for f in os.listdir(fallback_dir) if f.endswith('_crop_face.png'))
        
        all_face_files = sorted(primary_files | fallback_files)
        
        # 逐个加载：根据 prefer_fallback 决定优先级
        ref_images = []
        ref_frame_indices = []
        for face_file in all_face_files:
            face_img = None
            primary_path = os.path.join(primary_dir, face_file)
            fallback_path = os.path.join(fallback_dir, face_file)
            
            if self.prefer_fallback:
                # 优先从 fallback 读取
                if face_file in fallback_files:
                    try:
                        face_img = Image.open(fallback_path).convert("RGB")
                    except Exception:
                        pass  # fallback 损坏，尝试 primary
                
                # fallback 失败，尝试 primary
                if face_img is None and face_file in primary_files:
                    try:
                        face_img = Image.open(primary_path).convert("RGB")
                    except Exception:
                        pass  # primary 也失败
            else:
                # 优先从 primary 读取
                if face_file in primary_files:
                    try:
                        face_img = Image.open(primary_path).convert("RGB")
                    except Exception:
                        pass  # primary 损坏，尝试 fallback
                
                # primary 失败，尝试 fallback
                if face_img is None and face_file in fallback_files:
                    try:
                        face_img = Image.open(fallback_path).convert("RGB")
                    except Exception:
                        pass  # fallback 也失败
            
            # 成功读取则添加
            if face_img is not None:
                face_img = self.resize_ref(face_img, self.height, self.width)
                face_img = self.apply_mae_mask(face_img)
                ref_images.append(face_img)
                match = re.match(r"^(\d+)_crop_face\.png$", face_file)
                if match:
                    ref_frame_indices.append(int(match.group(1)))
                else:
                    # Fallback when filename does not follow expected pattern
                    ref_frame_indices.append(-1)

        # 检查人脸图片数量，不足3张则跳过该样本
        if len(ref_images) < self.ref_num:
            print(f"⚠️ 参考图片数量不足 ({len(ref_images)}/{self.ref_num})，跳过该样本: {video_path}")
            return None, None, None, None, None
        
        # 超过3张则截取前3张
        if len(ref_images) > self.ref_num:
            ref_images = ref_images[:self.ref_num]
            ref_frame_indices = ref_frame_indices[:self.ref_num]

        # ============================================================
        # [NEW] 获取参考图片对应的 pose 信息
        # ============================================================
        ref_poses = []
        if facedetect_v1 is not None and facedetect_v1_frame_index is not None and len(facedetect_v1) > 0 and len(facedetect_v1_frame_index) > 0:
            for frame_idx in ref_frame_indices:
                pose = self._get_pose_for_frame_index(facedetect_v1, facedetect_v1_frame_index, frame_idx)
                if pose is None:
                    # 没有找到精确匹配的 pose，使用默认值
                    ref_poses.append({'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0})
                else:
                    ref_poses.append(pose)
        else:
            # 如果没有 pose 数据，使用默认值
            ref_poses = [{'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}] * len(ref_frame_indices)

        # ============================================================
        # [DEPRECATED] 原有从视频裁剪人脸的逻辑 - 已注释
        # ============================================================

        return frames, ref_images, ref_frame_indices, sample_ids, ref_poses

    def __getitem__(self, index):
        max_retry = 10   # 最多重试 10 次，避免死循环
        retry = 0

        while retry < max_retry:
            # ----- 选择 train / test 数据 -----
            if self.training:
                meta_data = self.data_train[index % len(self.data_train)]
            else:
                meta_data = self.data_test[index % len(self.data_test)]

            video_path = meta_data['video_path']
            meta_prompt = meta_data['meta_prompt']
            condition_path = meta_data['condition_path']
            min_index = meta_data['min_index']
            max_index = meta_data['max_index']
            fps = meta_data['fps']
            facedetect_v1 = meta_data.get('facedetect_v1', [])  # [NEW] 获取 pose 数据
            facedetect_v1_frame_index = meta_data.get('facedetect_v1_frame_index', [])  # [NEW] 获取帧索引

            # ----- 尝试读取 video + ref -----
            try:
                input_video, ref_images, ref_frame_indices, video_frame_indices, ref_poses = self.load_video_crop_ref_image(
                    video_path=video_path,
                    condition_path=condition_path,
                    min_index=min_index,
                    max_index=max_index,
                    fps=fps,
                    facedetect_v1=facedetect_v1,  # [NEW] 传递 pose 数据
                    facedetect_v1_frame_index=facedetect_v1_frame_index  # [NEW] 传递帧索引
                )
            except Exception as e:
                print("❌ Exception in load_video_crop_ref_image")
                print(f"   video_path: {video_path}")
                print(f"   error type: {type(e).__name__}")
                print(f"   error msg : {e}")

                # 打印 traceback，定位问题更容易
                import traceback
                traceback.print_exc()
                input_video = None
                ref_images = None
                ref_frame_indices = None
                video_frame_indices = None
                ref_poses = None  # [NEW]
            # ----- 如果成功，并且 video 不为空，返回结果 -----
            if input_video is not None and len(input_video) > 0:
                return {
                    "global_caption": None,
                    "shot_num": 1,
                    "pre_shot_caption": [],
                    "single_caption": meta_prompt["single_prompt"],
                    "video": input_video,
                    "ref_num": len(ref_images),
                    "ref_images": ref_images,
                    "ref_frame_indices": ref_frame_indices,
                    "ref_poses": ref_poses,  # [NEW] 添加 pose 信息
                    "video_frame_indices": video_frame_indices,
                    "video_path": video_path
                }

            # ----- 如果失败，换 index，并继续尝试 -----
            retry += 1
            index = random.randint(0, len(self.data_train) - 1 if self.training else len(self.data_test) - 1)

        # 若 10 次都失败，返回最后一次的错误内容
        raise RuntimeError(f"❌ [Dataset] Failed to load video/ref after {max_retry} retries.")

    def __len__(self):
        if self.training:
            return len(self.data_train)
        else:
            return len(self.data_test)

if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    # ============================================================
    # 多线程预计算：验证所有视频的人脸图片有效性
    # ============================================================
    
    # 配置
    DATA_PATH = '/user/kg-aigc/rd_dev/jiahaochen/all_videos_wangpan_duration150.data'
    CONDITION_BASE_PATH = '/user/kg-aigc/rd_dev/jiahaochen/grounded_sam_wangpan_hq'
    CONDITION_FALLBACK_PATH = '/user/kg-aigc/rd_dev/jiahaochen/face_duration150'
    OUTPUT_JSON_PATH = '/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/conf/filterd_useable.json'
    REF_NUM = 3
    NUM_WORKERS = 32  # 线程数
    
    print(f"📂 数据源: {DATA_PATH}")
    print(f"📂 Primary目录: {CONDITION_BASE_PATH}")
    print(f"📂 Fallback目录: {CONDITION_FALLBACK_PATH}")
    print(f"📂 输出文件: {OUTPUT_JSON_PATH}")
    print(f"🔢 需要的参考图片数: {REF_NUM}")
    print(f"🧵 线程数: {NUM_WORKERS}")
    
    # 读取元数据
    print("\n📖 读取元数据...")
    meta_datas_list = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '{':
            for line in f:
                line = line.strip()
                if line:
                    try:
                        meta_datas_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        elif first_char == '[':
            meta_datas_list = json.load(f)
        else:
            f.seek(0)
            meta_datas_dict = json.load(f)
            meta_datas_list = list(meta_datas_dict.values())
    
    print(f"✅ 读取到 {len(meta_datas_list)} 条元数据")
    
    # 过滤出 mp4 文件
    mp4_list = [ctx for ctx in meta_datas_list if ctx.get("disk_path", "").lower().endswith(".mp4")]
    print(f"✅ 过滤出 {len(mp4_list)} 条 mp4 视频")
    
    # 验证函数（单个视频）
    def validate_video(context):
        """验证单个视频的人脸图片是否有效，返回 (context, is_valid, valid_count)"""
        disk_path = context.get("disk_path", "")
        # 规范化路径，处理双斜杠 // 等问题
        disk_path = os.path.normpath(disk_path)
        video_filename = os.path.basename(disk_path)
        condition_path_primary = os.path.join(CONDITION_BASE_PATH, video_filename)
        condition_path_fallback = os.path.join(CONDITION_FALLBACK_PATH, video_filename)
        
        # 检查目录是否存在
        if not os.path.isdir(condition_path_primary) and not os.path.isdir(condition_path_fallback):
            return context, False, 0
        
        # 合并两个目录的人脸图片文件名
        primary_files = set()
        fallback_files = set()
        if os.path.isdir(condition_path_primary):
            try:
                primary_files = set(f for f in os.listdir(condition_path_primary) if f.endswith('_crop_face.png'))
            except Exception:
                pass
        if os.path.isdir(condition_path_fallback):
            try:
                fallback_files = set(f for f in os.listdir(condition_path_fallback) if f.endswith('_crop_face.png'))
            except Exception:
                pass
        
        all_face_files = sorted(primary_files | fallback_files)
        
        # 验证图片是否能成功打开
        valid_face_count = 0
        for face_file in all_face_files:
            primary_path = os.path.join(condition_path_primary, face_file)
            fallback_path = os.path.join(condition_path_fallback, face_file)
            
            img_valid = False
            if face_file in primary_files:
                try:
                    img = Image.open(primary_path)
                    img.verify()
                    img_valid = True
                except Exception:
                    pass
            
            if not img_valid and face_file in fallback_files:
                try:
                    img = Image.open(fallback_path)
                    img.verify()
                    img_valid = True
                except Exception:
                    pass
            
            if img_valid:
                valid_face_count += 1
                if valid_face_count >= REF_NUM:
                    break
        
        is_valid = valid_face_count >= REF_NUM
        return context, is_valid, valid_face_count
    
    # 多线程验证
    print(f"\n🚀 开始多线程验证 ({NUM_WORKERS} 线程)...")
    valid_contexts = []
    invalid_count = 0
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(validate_video, ctx): ctx for ctx in mp4_list}
        
        with tqdm(total=len(mp4_list), desc="验证进度") as pbar:
            for future in as_completed(futures):
                try:
                    context, is_valid, valid_count = future.result()
                    if is_valid:
                        with lock:
                            valid_contexts.append(context)
                    else:
                        with lock:
                            invalid_count += 1
                except Exception as e:
                    with lock:
                        invalid_count += 1
                pbar.update(1)
    
    print(f"\n✅ 验证完成！")
    print(f"   有效视频: {len(valid_contexts)}")
    print(f"   无效视频: {invalid_count}")
    
    # 保存到 JSON 文件
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(valid_contexts, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 已保存到: {OUTPUT_JSON_PATH}")
    print(f"   文件大小: {os.path.getsize(OUTPUT_JSON_PATH) / 1024 / 1024:.2f} MB")
    
    
