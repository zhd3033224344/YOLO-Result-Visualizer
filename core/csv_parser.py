import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Ax3D
import os


def parse_yolo_csv(csv_path):
    """解析YOLO results.csv"""
    try:
        df = pd.read_csv(csv_path)
        # 清洗列名（去除空格）
        df.columns = [col.strip() for col in df.columns]
        return df
    except Exception as e:
        print(f"CSV解析失败：{e}")
        return None


def plot_3d_loss_curve(df, save_path):
    """绘制 损失函数3D图：Epoch × 训练损失 × 验证损失"""
    if df is None: return

    epoch = df['epoch']
    train_loss = df['train/box_loss'] + df['train/cls_loss']  # 总训练损失
    val_loss = df['val/box_loss'] + df['val/cls_loss']  # 总验证损失

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(epoch, train_loss, val_loss, lw=2, color='#FF4500', label='Loss Trend')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train Loss', fontsize=12)
    ax.set_zlabel('Val Loss', fontsize=12)
    ax.set_title('YOLO Loss 3D Trend Curve', fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("✅ 损失函数3D图已生成")


def get_csv_metrics(df):
    """提取核心指标，用于HTML报告"""
    if df is None: return {}
    last = df.iloc[-1]
    return {
        "best_epoch": int(df['epoch'].iloc[df['val/mAP50-95'].idxmax()]),
        "last_epoch": int(last['epoch']),
        "mAP50": round(last['val/mAP50'], 4),
        "mAP50-95": round(last['val/mAP50-95'], 4),
        "final_train_loss": round(last['train/box_loss'] + last['train/cls_loss'], 4),
        "final_val_loss": round(last['val/box_loss'] + last['val/cls_loss'], 4)
    }