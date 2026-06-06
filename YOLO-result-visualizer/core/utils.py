import os
import cv2

def yolo2xyxy(label_line, img_width, img_height):
    """YOLO格式 → 像素坐标"""
    class_id, x, y, w, h = map(float, label_line[:5])
    x1 = int((x - w / 2) * img_width)
    y1 = int((y - h / 2) * img_height)
    x2 = int((x + w / 2) * img_width)
    y2 = int((y + h / 2) * img_height)
    conf = float(label_line[5]) if len(label_line) > 5 else 1.0
    return int(class_id), x1, y1, x2, y2, conf

def get_image_label_pairs(img_dir, label_dir):
    """自动匹配图片+标签"""
    pairs = []
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")
    for img_name in os.listdir(img_dir):
        if img_name.lower().endswith(img_exts):
            base = os.path.splitext(img_name)[0]
            label_path = os.path.join(label_dir, f"{base}.txt")
            if os.path.exists(label_path):
                pairs.append((os.path.join(img_dir, img_name), label_path, img_name))
    return pairs

def get_gt_pred_pairs(img_dir, gt_dir, pred_dir):
    """自动匹配 图片+真值标签+预测标签（对比专用）"""
    pairs = []
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")
    for img_name in os.listdir(img_dir):
        if img_name.lower().endswith(img_exts):
            base = os.path.splitext(img_name)[0]
            gt_path = os.path.join(gt_dir, f"{base}.txt")
            pred_path = os.path.join(pred_dir, f"{base}.txt")
            if os.path.exists(gt_path) and os.path.exists(pred_path):
                pairs.append((os.path.join(img_dir, img_name), gt_path, pred_path, img_name))
    return pairs