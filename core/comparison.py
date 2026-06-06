import cv2
import numpy as np
from configs.class_names import CLASS_NAMES, COLORS
from core.utils import yolo2xyxy


def draw_single(img_path, label_path):
    """绘制单张图（对比内部调用）"""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().split()
        if len(line) < 5: continue
        cid, x1, y1, x2, y2, conf = yolo2xyxy(line, w, h)
        color = COLORS[cid]
        text = f"{CLASS_NAMES[cid]} {conf:.2f}" if conf < 1 else CLASS_NAMES[cid]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def compare_gt_pred(img_path, gt_path, pred_path, save_path):
    """生成 真值 VS 预测 对比图"""
    gt_img = draw_single(img_path, gt_path)
    pred_img = draw_single(img_path, pred_path)
    # 左右拼接
    combined = np.hstack((gt_img, pred_img))
    # 添加标题
    cv2.putText(combined, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(combined, "Prediction", (gt_img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(save_path, combined)


def batch_compare(img_dir, gt_dir, pred_dir, save_dir):
    """批量对比可视化"""
    from core.utils import get_gt_pred_pairs
    os.makedirs(save_dir, exist_ok=True)
    pairs = get_gt_pred_pairs(img_dir, gt_dir, pred_dir)

    for img_path, gt_path, pred_path, img_name in pairs:
        save_path = os.path.join(save_dir, img_name)
        compare_gt_pred(img_path, gt_path, pred_path, save_path)

    print(f"✅ 批量对比完成，共处理 {len(pairs)} 张图片")