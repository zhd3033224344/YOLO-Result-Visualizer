import matplotlib.pyplot as plt
import pandas as pd
import os
from configs.class_names import CLASS_NAMES
from core.utils import get_image_label_pairs


def generate_statistics(img_dir, label_dir, save_dir):
    """生成YOLO检测结果统计图表"""
    os.makedirs(save_dir, exist_ok=True)
    pairs = get_image_label_pairs(img_dir, label_dir)

    class_counts = {cls: 0 for cls in CLASS_NAMES}
    conf_list = []

    for _, label_path in pairs:
        with open(label_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) >= 5:
                cid = int(line[0])
                class_counts[CLASS_NAMES[cid]] += 1
                if len(line) > 5: conf_list.append(float(line[5]))

    # 1. 类别分布饼图
    plt.figure(figsize=(10, 6))
    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
    plt.title("YOLO 检测类别分布")
    plt.savefig(os.path.join(save_dir, "class_dist.png"), dpi=300)
    plt.close()

    # 2. 目标数量统计
    pd.Series(class_counts).plot(kind='bar', figsize=(12, 6))
    plt.title("YOLO 目标数量统计")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "count_dist.png"), dpi=300)
    print("✅ 统计图表生成完成！")