# 自定义你的YOLO类别名称 + 随机颜色
CLASS_NAMES = [
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches",

]
# 生成对应类别颜色（RGB）
import random
random.seed(42)
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in CLASS_NAMES]