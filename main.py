import argparse
import os
from core.visualizer import batch_visualize
from core.stats import generate_statistics
from core.comparison import batch_compare
from core.csv_parser import parse_yolo_csv, plot_3d_loss_curve, get_csv_metrics
from core.html_report import generate_html_report

def main():
    parser = argparse.ArgumentParser(description="YOLO Result Visualizer | 升级版")
    parser.add_argument("--img-dir", required=True, help="图片文件夹")
    parser.add_argument("--label-dir", help="普通标签文件夹（可视化用）")
    parser.add_argument("--gt-label-dir", help="真值标签文件夹（对比用）")
    parser.add_argument("--pred-label-dir", help="预测标签文件夹（对比用）")
    parser.add_argument("--csv-path", help="yolo262/results.csv 路径（损失3D图）")
    parser.add_argument("--save-dir", default="outputs", help="输出目录")
    args = parser.parse_args()

    # 创建输出文件夹
    os.makedirs(args.save_dir, exist_ok=True)
    metrics = {}
    has_compare = False
    has_csv = False

    # 1. 普通可视化
    if args.label_dir:
        batch_visualize(args.img_dir, args.label_dir, f"{args.save_dir}/visuals")
        generate_statistics(args.img_dir, args.label_dir, f"{args.save_dir}/stats")

    # 2. 真值VS预测对比
    if args.gt_label_dir and args.pred_label_dir:
        batch_compare(args.img_dir, args.gt_label_dir, args.pred_label_dir, f"{args.save_dir}/comparisons")
        has_compare = True

    # 3. CSV解析 + 损失3D图
    if args.csv_path:
        df = parse_yolo_csv(args.csv_path)
        if df is not None:
            plot_3d_loss_curve(df, os.path.join(args.save_dir, "loss_3d.png"))
            metrics = get_csv_metrics(df)
            has_csv = True

    # 4. 生成HTML总报告
    generate_html_report(args.save_dir, metrics, has_compare, has_csv)
    print("\n🎉 所有任务完成！打开 report.html 查看完整报告")

if __name__ == "__main__":
    main()