import json

input_file = "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/datasets/all_videos_wangpan.data"
output_file = "/root/paddlejob/workspace/qizipeng/baidu/personal-code/Multi-view/multi_view/datasets/videos_duration_5_7.json"

filtered_data = []
total_count = 0
filtered_count = 0

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        total_count += 1
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            # 尝试从 raw_meta.sub_meta[0].duration 获取 duration
            duration = None
            if 'raw_meta' in data and 'sub_meta' in data['raw_meta']:
                sub_meta = data['raw_meta']['sub_meta']
                if sub_meta and len(sub_meta) > 0 and 'duration' in sub_meta[0]:
                    duration = sub_meta[0]['duration']
            
            # 如果没有找到，尝试从 clip_end - clip_start 计算
            if duration is None and 'raw_meta' in data:
                clip_end = data['raw_meta'].get('clip_end')
                clip_start = data['raw_meta'].get('clip_start')
                if clip_end is not None and clip_start is not None:
                    duration = clip_end - clip_start
            
            # 筛选 duration 在 5-7 秒之间的数据
            if duration is not None and 5 <= duration <= 7:
                filtered_data.append(data)
                filtered_count += 1
                
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            continue

# 保存为 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"总记录数: {total_count}")
print(f"筛选后记录数: {filtered_count}")
print(f"结果已保存到: {output_file}")