import cv2
from configs.class_names import CLASS_NAMES, COLORS
from core.utils import yolo2xyxy


def draw_yolo_result(img_path, label_path, save_path):
    """
    单张图片YOLO结果可视化
    """
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 读取标签
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split()
        if len(line) != 5:
            continue
        label_line = list(map(float, line))
        class_id, x1, y1, x2, y2 = yolo2xyxy(label_line, w, h)
        color = COLORS[class_id]
        class_name = CLASS_NAMES[class_id]
        conf = label_line[5] if len(label_line) > 5 else 1.0  # 置信度

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 标注文字
        text = f"{class_name} {conf:.2f}" if conf < 1 else class_name
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(save_path, img)
    return img


def batch_visualize(img_dir, label_dir, save_dir):
    """批量可视化"""
    from core.utils import get_image_label_pairs
    import os
    os.makedirs(save_dir, exist_ok=True)
    pairs = get_image_label_pairs(img_dir, label_dir)

    for img_path, label_path in pairs:
        save_name = os.path.basename(img_path)
        save_path = os.path.join(save_dir, save_name)
        draw_yolo_result(img_path, label_path, save_path)
    print(f"✅ 批量可视化完成，共处理 {len(pairs)} 张图片，保存至：{save_dir}")