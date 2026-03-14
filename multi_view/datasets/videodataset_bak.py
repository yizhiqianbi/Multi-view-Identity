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
    def __init__(self, dataset_base_path='/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/datasets/merged_mark_paishe_ds_meg_merge_dwposefilter_paishe.json',
                        ref_image_path='/root/paddlejob/workspace/qizipeng/code/longvideogen/output.json',
                        time_division_factor=4, 
                        time_division_remainder=1,
                        max_pixels=1920*1080,
                        height_division_factor=16, width_division_factor=16,
                        transform=None,
                        length=None,
                        resolution=None,
                        prev_length=5,
                        ref_num = 3,
                        training = True):
        self.data_path = dataset_base_path
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
            if not disk_path.lower().endswith(".mp4"):
                continue


            # reader = imageio.get_reader(meta_datas[video_path]["disk_path"])
            # total_original_frames = reader.count_frames()
            # total_frame = total_original_frames # context["end_index"] - context["start_index"] - 1
            total_frame = None
            ref_id  = self.get_ref_id(face_crop_angle = context['facedetect_v1'], facedetect_v1_frame_index = context['facedetect_v1_frame_index'], total_frame = total_frame)
            if ref_id == []:
                continue
            ref_id_all = []
            for ids in ref_id:
                ref_id_grop = []
                for id in ids:
                    coordinate = context['facedetect_v1'][id][0]['detect']
                    if context['facedetect_v1'][id][0]['detect']["prob"] < 0.99:
                        continue
                    top, height, width, left = coordinate['top'], coordinate['height'], coordinate['width'], coordinate['left']
                    if not(min(height, width) > 80 ):
                        continue
                    # enlarge bbox 1.5x
                    width = int(width * 1)
                    height = int(height * 1)
                    frame_index = context['facedetect_v1_frame_index'][id]
                    ref_id_grop.append([top, height, width, left, int(frame_index)])
                if ref_id_grop != []:
                    if len(ref_id_grop) >= 3: #self.ref_num: ### 为了和ref_num = 3 保持数据一致
                        ref_id_all.append(ref_id_grop)
            if ref_id_all == []:
                continue
            meta_prompt = {}
            meta_prompt["global_caption"] = None
            meta_prompt["per_shot_prompt"] = []
            meta_prompt["single_prompt"] = context['text']
            self.data.append({'video_path': disk_path, 'meta_prompt': meta_prompt, "ref_id_all": ref_id_all})
            # self.data.append({'video_path':video_path, 'meta_prompt': meta_prompt, "ref_id_all": ref_id_all})

        random.seed(42)  # 让每次划分一致（可选）
        total = len(self.data)
        test_count = max(1, int(total * 0.05))  # 至少一个

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

    def get_ref_id(self, face_crop_angle, facedetect_v1_frame_index = None, total_frame = None, angle_threshold=50):
        """
            返回满足角度差异要求的三元组 [i, j, k]
            要求：
                - face_crop_angle[i] / [j] / [k] 都必须非空
                - i,j 两者任意 yaw/pitch/roll 差值 > angle_threshold
                - k != i != j，且 k 也必须非空
            """
        ref_id = []
        max_try = 5
        need_max = 3
        try_num = 0
            
        # 过滤空元素，保留有效索引
        valid_indices = [idx for idx, item in enumerate(face_crop_angle) if item]
        N = len(valid_indices)

        if N < 3:
            return ref_id   # 不足 3 张有效图，无法组成三元组

        # 两两组合检查角度差
        for a in range(N - 1):
            i = valid_indices[a]
            # if facedetect_v1_frame_index[i] > total_frame:
            #     continue
            angle_i = face_crop_angle[i][0]["angle"]

            for b in range(a + 1, N):
                j = valid_indices[b]
                # if facedetect_v1_frame_index[j] > total_frame:
                #     continue
                angle_j = face_crop_angle[j][0]["angle"]

                # 判断是否满足阈值
                if (
                    abs(angle_i["yaw"]   - angle_j["yaw"])   > angle_threshold or
                    abs(angle_i["pitch"] - angle_j["pitch"]) > angle_threshold or
                    abs(angle_i["roll"]  - angle_j["roll"])  > angle_threshold
                ):
                    # 找第三个 k
                    for c in range(N):
                        k = valid_indices[c]
                        # if facedetect_v1_frame_index[k] > total_frame:
                        #     continue
                        if k != i and k != j:
                            ref_id.append([i, j, k])
                            break

                    try_num += 1
                    if try_num >= max_try or len(ref_id) >= need_max:
                        return ref_id

        return ref_id
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


    def load_video_crop_ref_image(self, video_path=None, ref_id_all=[[]]):
        ### fps 转化
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        original_fps = meta.get("fps", 24)
        target_fps = 16
        duration_seconds = 5
        target_frames = target_fps * duration_seconds + 1   # = 80 frames

        # ---- 获取原视频帧数 ----
        try:
            total_original_frames = reader.count_frames()
        except:
            total_original_frames = int(meta.get("duration", 5) * original_fps)



        # ---- 需要多少原始帧（5秒）----
        need_orig_frames = int(original_fps * duration_seconds)

        # ---- Case 1: 原视频 >= 5秒 → 随机选择 5 秒起点 ----
        if total_original_frames > need_orig_frames:
            max_start = total_original_frames - need_orig_frames
            start_frame = random.randint(0, max_start)
            segment_start = start_frame
            segment_end = start_frame + need_orig_frames
        else:
            # ---- Case 2: 原视频 < 5秒 → 用全部帧 ----
            segment_start = 0
            segment_end = total_original_frames

        # ---- 均匀采样 80 帧 ----
        sample_ids = np.linspace(segment_start, segment_end - 1, num=target_frames, dtype=int)

        frames = []
        for frame_id in sample_ids:
            frame = reader.get_data(int(frame_id))
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)

        # ===========================
        #  选择参考图部分（你要求的）
        # ===========================

        # 1）从 ref_images_all（三维 list）里随机选一组
        #    ref_images_all = [ [img1, img2, img3], [imgA, imgB, imgC], ... ]
        ref_group = random.choice(ref_id_all)

        # 2）检查资源是否足够
        if len(ref_group) < self.ref_num:
            raise ValueError(f"需要 {self.ref_num} 张参考图，但该组只有 {len(ref_group)} 张。")

        # 3）从该组中随机选 self.ref_num 张
        selected_refs = random.sample(ref_group, self.ref_num)
        random.shuffle(selected_refs)
        
        ref_images = []
        for sf in selected_refs:
            top, height, width, left, frame_index = sf
            # import pdb; pdb.set_trace()
            if frame_index > total_original_frames:
                print(f"{video_path}, frame_index({frame_index}) out of range")
            frame = reader.get_data(int(frame_index))
            frame = Image.fromarray(frame)
            xmin, ymin, xmax, ymax = left, top, left + width, top + height
            cropped_image = frame.crop((xmin, ymin, xmax, ymax)).convert("RGB")
            cropped_image = self.resize_ref(cropped_image, self.height, self.width)
            # Calculate the required size to keep aspect ratio and fill the rest with padding.
            ref_images.append(cropped_image)
        reader.close()

        return frames, ref_images

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
            ref_id_all = meta_data['ref_id_all']

            # ----- 尝试读取 video + ref -----
            try:
                input_video, ref_images = self.load_video_crop_ref_image(
                    video_path=video_path,
                    ref_id_all=ref_id_all
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
            # ----- 如果成功，并且 video 不为空，返回结果 -----
            if input_video is not None and len(input_video) > 0:
                return {
                    "global_caption": None,
                    "shot_num": 1,
                    "pre_shot_caption": [],
                    "single_caption": meta_prompt["single_prompt"],
                    "video": input_video,
                    "ref_num": self.ref_num,
                    "ref_images": ref_images,
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
    from torch.utils.data import DataLoader
    dataset = MulltiShot_MultiView_Dataset(length=49, resolution=(384, 640), training=True)
    print(len(dataset))
    metadata = dataset[0]
    # results = dataset[0]
    # loader = DataLoader(
    #     dataset,
    #     batch_size=1,          # 视频一般 batch=1
    #     shuffle=False,         # 你想打乱就 True
    #     num_workers=10,         # ⭐ 重点：开启 8 个子进程并行加载
    #     pin_memory=True,
    #     prefetch_factor=2,     # 每个 worker 预读取 2 个样本
    #     collate_fn=lambda x: x[0],   # ⭐ 不做任何 collate
    # )

    # for batch in tqdm(loader):
    #     pass
    for i in tqdm(range(len(dataset))):
        file = dataset[i]

    assert 0
    