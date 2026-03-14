import imageio, os, torch, warnings, torchvision, argparse, json
from PIL import Image
import pandas as pd
from tqdm import tqdm
import random
from decord import VideoReader
from decord import cpu, gpu
import imageio.v3 as iio
import glob

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
    """
    电影数据集类，用于加载电影镜头+主角人脸数据
    与原 MulltiShot_MultiView_Dataset 返回格式完全对齐，实现即插即用
    """
    # 预计算文件路径（类变量）
    PRECOMPUTED_JSON_PATH = '/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/conf/filterd_movie_useable.json'
    
    # 主角ID列表
    MAIN_ACTOR_IDS = ['主角1', '主角2', '主角3']
    
    def __init__(self, 
                 dataset_base_path='/user/kg-aigc/rd_dev/jiahaochen/all_videos_large_angle/',
                 face_base_path='/user/kg-aigc/rd_dev/jiahaochen/grounded_sam_results_hq',
                 video_prefix='/root/digital_data/',
                 time_division_factor=4, 
                 time_division_remainder=1,
                 max_pixels=1920*1080,
                 height_division_factor=16, width_division_factor=16,
                 transform=None,
                 length=None,
                 resolution=None,
                 prev_length=5,
                 ref_num=3,
                 training=True,
                 precomputed_json_path=None):
        
        self.dataset_base_path = dataset_base_path
        self.face_base_path = face_base_path
        self.video_prefix = video_prefix
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
        
        # 确定预计算文件路径
        if precomputed_json_path is not None:
            self.precomputed_json_path = precomputed_json_path
        else:
            self.precomputed_json_path = self.PRECOMPUTED_JSON_PATH

        # ============================================================
        # 优先加载预计算的 JSON 文件，跳过耗时验证
        # ============================================================
        if os.path.exists(self.precomputed_json_path):
            print(f"✅ 发现预计算文件，直接加载: {self.precomputed_json_path}")
            with open(self.precomputed_json_path, 'r', encoding='utf-8') as f:
                precomputed_data = json.load(f)
            print(f"✅ 从预计算文件加载了 {len(precomputed_data)} 条数据")
            
            # 快速构建 self.data（无需验证，预计算已验证过）
            for item in tqdm(precomputed_data, desc="加载预计算数据"):
                self.data.append(item)
        else:
            # ============================================================
            # 预计算文件不存在，从原始数据加载并验证
            # ============================================================
            print(f"⚠️ 预计算文件不存在: {self.precomputed_json_path}")
            print(f"📖 将从原始数据加载并验证...")
            
            self._load_and_validate_data()

        # 划分训练集和测试集
        random.seed(42)
        total = len(self.data)
        test_count = min(20, total)  # 固定20个测试样本

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

    def _load_and_validate_data(self):
        """从原始 .data 文件加载并验证数据"""
        # 读取目录下所有 .data 文件
        data_files = glob.glob(os.path.join(self.dataset_base_path, '*.data'))
        print(f"📂 发现 {len(data_files)} 个数据文件")
        
        for data_file in tqdm(data_files, desc="读取数据文件"):
            # 从文件名提取电影编号
            movie_id = os.path.basename(data_file).replace('.data', '')
            
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        context = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # 筛选仅包含主角的记录
                    ids = context.get('ID', [])
                    main_actor = None
                    for actor_id in ids:
                        if actor_id in self.MAIN_ACTOR_IDS:
                            main_actor = actor_id
                            break
                    
                    if main_actor is None:
                        continue  # 跳过没有主角的记录
                    
                    # 构建人脸目录路径
                    face_dir = self._get_face_dir(movie_id, main_actor)
                    if face_dir is None:
                        continue  # 跳过没有人脸数据的记录
                    
                    # 验证人脸图片数量
                    face_files = self._get_face_files(face_dir)
                    if len(face_files) < self.ref_num:
                        continue  # 跳过人脸图片不足的记录
                    
                    # 构建视频路径
                    video_mount_path = context.get('video_mount_path', '')
                    video_path = os.path.join(self.video_prefix, video_mount_path)
                    
                    # 获取caption
                    caption = context.get('caption_minicpm45_sft_gpu_v5_3', '')
                    
                    # 获取视频时长
                    video_duration = context.get('video_duration', 5.0)
                    
                    meta_prompt = {
                        "global_caption": None,
                        "per_shot_prompt": [],
                        "single_prompt": caption
                    }
                    
                    self.data.append({
                        'video_path': video_path,
                        'meta_prompt': meta_prompt,
                        'face_dir': face_dir,
                        'movie_id': movie_id,
                        'main_actor': main_actor,
                        'video_duration': video_duration,
                        'identity': context.get('identity', ''),
                        'raw_context': context  # 保存原始数据用于调试
                    })
        
        print(f"✅ 验证完成，有效数据: {len(self.data)} 条")

    def _get_face_dir(self, movie_id, actor_id):
        """
        获取人脸图片目录路径
        优先使用"三个视角"目录，不存在时回退到"所有视角"
        """
        base_dir = os.path.join(self.face_base_path, f'mov{movie_id}', actor_id)
        
        # 优先使用"三个视角"目录
        three_view_dir = os.path.join(base_dir, '三个视角')
        if os.path.isdir(three_view_dir):
            return three_view_dir
        
        # 回退到"所有视角"目录
        all_view_dir = os.path.join(base_dir, '所有视角')
        if os.path.isdir(all_view_dir):
            return all_view_dir
        
        # 对于配角等没有子目录的情况，直接使用base_dir
        if os.path.isdir(base_dir):
            # 检查是否直接包含图片文件
            png_files = [f for f in os.listdir(base_dir) if f.endswith('.png')]
            if png_files:
                return base_dir
        
        return None

    def _get_face_files(self, face_dir):
        """获取目录下所有的人脸图片文件"""
        if not face_dir or not os.path.isdir(face_dir):
            return []
        
        face_files = sorted([
            os.path.join(face_dir, f) 
            for f in os.listdir(face_dir) 
            if f.endswith('.png')
        ])
        return face_files

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

    def resize_ref(self, img, target_h, target_w):
        h = target_h
        w = target_w
        img = img.convert("RGB")
        img_ratio = img.width / img.height
        target_ratio = w / h
        
        if img_ratio > target_ratio:
            new_width = w
            new_height = int(new_width / img_ratio)
        else:
            new_height = h
            new_width = int(new_height * img_ratio)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        delta_w = w - img.size[0]
        delta_h = h - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        new_img = ImageOps.expand(img, padding, fill=(255, 255, 255))

        return new_img

    def load_video_crop_ref_image(self, video_path=None, face_dir=None, video_duration=5.0):
        """
        加载视频帧和参考人脸图片
        
        Args:
            video_path: 视频文件路径
            face_dir: 人脸图片目录路径
            video_duration: 视频时长
        """
        # 读取视频
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        original_fps = meta.get("fps", 25.0)
        target_fps = 16
        duration_seconds = 5
        target_frames = target_fps * duration_seconds + 1   # = 81 frames

        # 获取原视频帧数
        try:
            total_original_frames = reader.count_frames()
        except:
            total_original_frames = int(meta.get("duration", video_duration) * original_fps)

        # 需要多少原始帧（5秒）
        need_orig_frames = int(original_fps * duration_seconds)
        
        # 从视频开头采样
        segment_start = 0
        segment_end = min(need_orig_frames, total_original_frames)
        available_frames = segment_end - segment_start

        # ============================================================
        # 帧数不足处理策略：
        # 1. 如果可用帧数 >= 目标帧数，正常均匀采样
        # 2. 如果可用帧数 < 目标帧数，先采样所有帧，然后用最后一帧填充
        # ============================================================
        frames = []
        sample_ids = []
        
        if available_frames >= target_frames:
            # 正常情况：均匀采样 81 帧
            sample_ids = np.linspace(segment_start, segment_end - 1, num=target_frames, dtype=int)
            for frame_id in sample_ids:
                frame = reader.get_data(int(frame_id))
                frame = Image.fromarray(frame)
                frame = self.crop_and_resize(frame, *self.get_height_width(frame))
                frames.append(frame)
        else:
            # 帧数不足：先读取所有可用帧，然后用最后一帧填充
            # 注：减少日志输出，避免分布式训练时刷屏
            # print(f"⚠️ 视频帧数不足 ({available_frames}/{target_frames})，使用尾帧填充: {video_path}")
            
            # 读取所有可用帧
            for frame_id in range(segment_start, segment_end):
                sample_ids.append(frame_id)
                frame = reader.get_data(int(frame_id))
                frame = Image.fromarray(frame)
                frame = self.crop_and_resize(frame, *self.get_height_width(frame))
                frames.append(frame)
            
            # 用最后一帧填充到目标帧数
            if len(frames) > 0:
                last_frame = frames[-1]
                padding_count = target_frames - len(frames)
                for _ in range(padding_count):
                    frames.append(last_frame.copy())
                    sample_ids.append(sample_ids[-1])
        
        reader.close()

        # 读取参考人脸图片
        face_files = self._get_face_files(face_dir)
        ref_images = []
        ref_frame_indices = []
        
        for face_file in face_files[:self.ref_num]:  # 只取前ref_num张
            try:
                face_img = Image.open(face_file).convert("RGB")
                face_img = self.resize_ref(face_img, self.height, self.width)
                ref_images.append(face_img)
                base_name = os.path.basename(face_file)
                match = re.search(r"(\\d+)", base_name)
                if match:
                    ref_frame_indices.append(int(match.group(1)))
                else:
                    ref_frame_indices.append(-1)
            except Exception as e:
                print(f"⚠️ 无法加载人脸图片: {face_file}, 错误: {e}")
                continue

        # 检查人脸图片数量
        if len(ref_images) < self.ref_num:
            print(f"⚠️ 参考图片数量不足 ({len(ref_images)}/{self.ref_num})，跳过该样本: {video_path}")
            return None, None, None, None

        return frames, ref_images, ref_frame_indices, sample_ids

    def __getitem__(self, index):
        max_retry = 10
        retry = 0

        while retry < max_retry:
            # 选择 train / test 数据
            if self.training:
                meta_data = self.data_train[index % len(self.data_train)]
            else:
                meta_data = self.data_test[index % len(self.data_test)]

            video_path = meta_data['video_path']
            meta_prompt = meta_data['meta_prompt']
            face_dir = meta_data['face_dir']
            video_duration = meta_data.get('video_duration', 5.0)

            # 尝试读取 video + ref
            try:
                input_video, ref_images, ref_frame_indices, video_frame_indices = self.load_video_crop_ref_image(
                    video_path=video_path,
                    face_dir=face_dir,
                    video_duration=video_duration
                )
            except Exception as e:
                print("❌ Exception in load_video_crop_ref_image")
                print(f"   video_path: {video_path}")
                print(f"   error type: {type(e).__name__}")
                print(f"   error msg : {e}")

                import traceback
                traceback.print_exc()
                input_video = None
                ref_images = None
                ref_frame_indices = None
                video_frame_indices = None
            
            # 如果成功，并且 video 不为空，返回结果
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
                    "video_frame_indices": video_frame_indices,
                    "video_path": video_path
                }

            # 如果失败，换 index，并继续尝试
            retry += 1
            index = random.randint(0, len(self.data_train) - 1 if self.training else len(self.data_test) - 1)

        raise RuntimeError(f"❌ [Dataset] Failed to load video/ref after {max_retry} retries.")

    def __len__(self):
        if self.training:
            return len(self.data_train)
        else:
            return len(self.data_test)
    
    def save_precomputed_json(self, output_path=None):
        """保存预计算数据到JSON文件"""
        if output_path is None:
            output_path = self.precomputed_json_path
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 预计算数据已保存到: {output_path}")
        print(f"   数据条数: {len(self.data)}")


if __name__ == '__main__':
    """
    测试功能：统计视频帧数分布，检测帧数不足的视频
    """
    import shutil
    from collections import defaultdict
    
    print("=" * 60)
    print("🎬 电影数据集 - 帧数统计")
    print("=" * 60)
    
    # 创建数据集实例
    dataset = MulltiShot_MultiView_Dataset(
        resolution=(480, 832),
        length=81,
        training=True
    )
    
    print(f"\n📊 数据集基本统计:")
    print(f"   总数据条数: {len(dataset.data)}")
    print(f"   训练集大小: {len(dataset.data_train)}")
    print(f"   测试集大小: {len(dataset.data_test)}")
    
    # ============================================================
    # 统计所有视频的帧数
    # ============================================================
    print(f"\n🔍 开始统计所有视频帧数...")
    
    target_fps = 16
    duration_seconds = 5
    target_frames = target_fps * duration_seconds + 1  # = 81 frames
    
    frame_stats = {
        'total': 0,
        'sufficient': 0,      # 帧数 >= 81
        'insufficient': 0,    # 帧数 < 81
        'frame_counts': [],   # 所有视频的可用帧数
        'insufficient_videos': []  # 帧数不足的视频详情
    }
    
    for i, meta_data in enumerate(tqdm(dataset.data, desc="统计帧数")):
        video_path = meta_data['video_path']
        video_duration = meta_data.get('video_duration', 5.0)
        
        try:
            reader = imageio.get_reader(video_path)
            meta = reader.get_meta_data()
            original_fps = meta.get("fps", 25.0)
            
            # 获取原视频帧数
            try:
                total_original_frames = reader.count_frames()
            except:
                total_original_frames = int(meta.get("duration", video_duration) * original_fps)
            
            reader.close()
            
            # 计算可用帧数（5秒内的帧数）
            need_orig_frames = int(original_fps * duration_seconds)
            available_frames = min(need_orig_frames, total_original_frames)
            
            frame_stats['total'] += 1
            frame_stats['frame_counts'].append(available_frames)
            
            if available_frames >= target_frames:
                frame_stats['sufficient'] += 1
            else:
                frame_stats['insufficient'] += 1
                frame_stats['insufficient_videos'].append({
                    'video_path': video_path,
                    'available_frames': available_frames,
                    'total_frames': total_original_frames,
                    'original_fps': original_fps,
                    'video_duration': video_duration
                })
                
        except Exception as e:
            print(f"⚠️ 无法读取视频: {video_path}, 错误: {e}")
            continue
    
    # ============================================================
    # 输出统计结果
    # ============================================================
    print(f"\n" + "=" * 60)
    print(f"📈 帧数统计结果")
    print(f"=" * 60)
    
    print(f"\n总体统计:")
    print(f"   检测视频数: {frame_stats['total']}")
    print(f"   帧数充足 (>= {target_frames}): {frame_stats['sufficient']} ({100*frame_stats['sufficient']/max(1,frame_stats['total']):.1f}%)")
    print(f"   帧数不足 (< {target_frames}): {frame_stats['insufficient']} ({100*frame_stats['insufficient']/max(1,frame_stats['total']):.1f}%)")
    
    if frame_stats['frame_counts']:
        frame_counts = np.array(frame_stats['frame_counts'])
        print(f"\n帧数分布:")
        print(f"   最小帧数: {frame_counts.min()}")
        print(f"   最大帧数: {frame_counts.max()}")
        print(f"   平均帧数: {frame_counts.mean():.1f}")
        print(f"   中位数帧数: {np.median(frame_counts):.1f}")
    
    # 按帧数区间统计
    if frame_stats['frame_counts']:
        print(f"\n帧数区间分布:")
        bins = [0, 20, 40, 60, 80, 81, 100, 150, float('inf')]
        bin_labels = ['0-20', '21-40', '41-60', '61-80', '81', '82-100', '101-150', '>150']
        for i in range(len(bins) - 1):
            count = sum(1 for f in frame_counts if bins[i] < f <= bins[i+1])
            if count > 0:
                print(f"   {bin_labels[i]}: {count} ({100*count/len(frame_counts):.1f}%)")
    
    # 列出帧数不足的视频
    if frame_stats['insufficient_videos']:
        print(f"\n⚠️ 帧数不足的视频详情 (前20个):")
        for i, v in enumerate(frame_stats['insufficient_videos'][:20]):
            print(f"   {i+1}. {os.path.basename(v['video_path'])}")
            print(f"      可用帧: {v['available_frames']}, 总帧: {v['total_frames']}, FPS: {v['original_fps']:.1f}, 时长: {v['video_duration']:.2f}s")
    
    # 保存完整统计到文件
    stats_output_path = '/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/conf/frame_stats.json'
    stats_to_save = {
        'total': frame_stats['total'],
        'sufficient': frame_stats['sufficient'],
        'insufficient': frame_stats['insufficient'],
        'target_frames': target_frames,
        'insufficient_videos': frame_stats['insufficient_videos']
    }
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
    print(f"\n📁 完整统计已保存到: {stats_output_path}")
    
    print(f"\n🎉 统计完成！")
