# YOLO-Result-Visualizer
🚀 一站式YOLO检测结果可视化&分析工具，独立Python程序，**不依赖YOLO训练环境**，批量绘图、真值预测对比、解析训练CSV、绘制3D损失曲线、自动生成静态HTML评测报告。

<p align="center">
<img src="./assets/demo_visual.jpg" width="420"/>
<img src="./assets/demo_compare.jpg" width="420"/>
</p>
<p align="center">
<img src="./assets/stats_chart.png" width="420"/>
<img src="./assets/loss_3d.png" width="420"/>
</p>

## 📖 项目介绍
面向YOLOv5/v8/v9/v10/Ultralytics全系列模型，针对数据集标签、模型推理结果做可视化分析；支持解析训练产出`results.csv`，自动统计指标、绘制损失三维曲线，最终打包生成可离线打开的HTML汇总报告，算法调参、数据集校验、项目汇报专用工具。

## 🎯 功能清单
- ✅ 批量可视化YOLO txt标签（数据集真值/模型预测框，自动配色+类别+置信度）
- ✅ GT&Pred左右拼接对比图（真值框vs预测框同屏对比，快速定位漏检、误检）
- ✅ 自动统计目标类别数量，生成饼图、柱状分布图
- ✅ 解析YOLO训练`results.csv`，提取mAP50/mAP50-95/最优epoch等关键指标
- ✅ Epoch-训练损失-验证损失三维可视化曲线图
- ✅ 一键生成**纯离线静态HTML报告**，集成所有图表、对比图、训练指标，无需联网即可打开分享
- ✅ 自定义数据集类别名称、标注颜色
- ✅ 全平台兼容：Windows/Linux/Mac，仅3项基础Python依赖

## 🛠️ 快速开始
### 1. 环境安装
```bash
# 克隆项目
git clone https://github.com/你的用户名/YOLO-Result-Visualizer.git
cd YOLO-Result-Visualizer

# 安装依赖
pip install -r requirements.txt