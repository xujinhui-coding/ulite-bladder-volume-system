# U-Lite

> 基于 [duong-db/U-Lite](https://github.com/duong-db/U-Lite) 开源代码实现的**"1M parameters are enough? A lightweight CNN-Based model for medical image segmentation"** 医学图像分割模型
>
> 本仓库在原作者代码基础上，增加了膀胱超声图像数据集支持、ONNX 部署导出、U-Lite + Snake 两阶段分割流水线等自定义功能。
>
> 一个用于医学图像分割的轻量级 CNN 模型（参数量 ~800K~1M），采用 U-Net 风格的编码器-解码器架构，核心亮点是使用**轴向深度可分离卷积**（Axial Depthwise Separable Convolution）替代标准卷积，大幅降低参数量。

---

## 目录结构

```
u-lite/
├── models/                     # 模型定义
│   ├── __init__.py             # 模型包入口
│   ├── common.py               # 基础网络模块（轴向卷积等）
│   └── ulite.py                # U-Lite 网络主体
├── utils/                      # 工具模块
│   ├── __init__.py             # 工具包入口
│   ├── tools.py                # 通用工具函数
│   ├── losses.py               # 损失函数（Dice Loss）
│   ├── metrics.py              # 评估指标（IoU, mAP, Precision/Recall/F1）
│   ├── yolo_datasets.py        # YOLO 格式数据集与数据增强
│   └── mask_datasets.py        # 掩膜数据集（未实现框架）
├── data/                       # 数据集
│   ├── chula/                  # 多目标大图数据集（~512×640）
│   ├── crop_chula/             # 单目标裁剪小图数据集（~64×64）
│   ├── dataset1_ulite/         # dataset1 训练数据集（256×256）
│   └── dataset3_ulite/         # dataset3 训练数据集（512×640）
├── runs/                       # 训练输出
│   └── dataset1_scratch/       # dataset1 模型训练结果
├── train.py                    # 训练主脚本
├── infer_demo.py               # 推理演示脚本
├── infer_single.py             # 单张图像推理（dataset/valid）
├── infer_onnx.py               # ONNX 推理（无需 PyTorch）
├── infer_dataset1.py           # dataset1/test 批量推理
├── infer_dataset1_valid.py     # dataset1/valid 推理 + 真值对比
├── infer_dataset1_pack.py      # dataset1 test 推理 + HTML 打包
├── infer_dataset3.py           # dataset3/test 批量推理
├── infer_dataset3_valid.py     # dataset3/valid 推理 + COCO 真值对比
├── export_onnx.py              # 模型导出为 ONNX 格式
├── ulite_snake_pipeline.py     # U-Lite + Snake 两阶段分割流水线
├── fileprocessing.py           # 数据预处理工具
├── test_datasets.py            # 数据集调试与可视化
├── requirements.txt            # 依赖
├── .gitattributes              # Git 属性配置
└── README.md                   # 本文档
```

---

## 模型 (`models/`)

### `models/__init__.py`
- 导出 `ULiteNet` 和 `createULite` 两个接口
- 外部通过 `from models import createULite` 即可创建模型

### `models/common.py`
定义了 U-Lite 的基础构件块：

| 组件 | 说明 |
|---|---|
| `AxialDWConv` | **轴向深度可分离卷积**：`x → x + DW_k×1(x) + DW_1×k(x)`，并行分支结构 |
| `MyAxialDWConv` | 串行版本：`x → x + DW_1×k(DW_k×1(x))` |
| `AxialDPConv` | 轴向深度可分离 + 逐点卷积组合：`AxialDW → BN → Conv1×1 → Act` |
| `AxialDWBottleNeck` | 瓶颈模块：`Conv1×1 → AxialDWConv → Conv1×1 → Act` |
| `autopad` | 自动计算 padding 使输出尺寸与输入一致 |
| `eval_act` | 激活函数解析器，支持字符串形式（如 `"nn.GELU()"`） |

### `models/ulite.py`
**U-Lite 网络主体**，采用 U 形编码器-解码器架构：

```
输入 → InConv → Encoder×5 → Bottleneck → Decoder×5 → OutConv → 输出
                │                    ↑           │
                └──── skip connections ──────────┘
```

- `ULiteNet` — 网络主类，支持自定义 `in_channels`、`mid_channels`（各层通道数）、`num_classes`、激活函数、是否使用串行版本
- `createULite(cfg)` — 从配置字典创建模型的工厂函数

---

## 训练 (`train.py`)

训练主脚本，支持命令行参数配置：

```bash
python train.py \
    --data data/dataset1_ulite \
    --img-size 256 256 \
    --cfg data/dataset1_ulite/param.yaml \
    --workers 0 \
    --epochs 300 \
    --batch-size 8 \
    --gpu 0 \
    --save-dir runs/dataset1_scratch
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data` | `data` | 数据目录（需包含 train.txt, valid.txt, classes.txt） |
| `--weights` | `""` | 预训练权重路径 |
| `--img-size` | `[640, 640]` | 输入图像尺寸 [height, width] |
| `--cfg` | `data/param.yaml` | 模型与训练配置 YAML |
| `--workers` | `4` | 数据加载进程数 |
| `--epochs` | `3` | 训练轮数 |
| `--batch-size` | `1` | 批次大小 |
| `--adam` | `false` | 使用 Adam 优化器（默认 SGD） |
| `--gpu` | `1` | GPU 设备号 |
| `--amp` | `false` | 混合精度训练 |
| `--cache` | — | 缓存图像到 RAM 或磁盘 |
| `--save-dir` | `runs/exp` | 结果保存目录 |
| `--seed` | `8888` | 随机种子 |

### 优化器与学习率

- **优化器**：SGD（默认）或 Adam（加 `--adam`）
- **学习率调度**：CosineAnnealingLR（余弦退火）
- **关键超参数**（从 `param.yaml` 读取）：
  - `lr0`：初始学习率
  - `momentum`：动量 / Adam beta1
  - `weight_decay`：权重衰减

### 损失函数

- `nn.BCELoss()` — 二值交叉熵损失
- `DiceLoss()` — Dice 损失
- **总损失 = BCELoss + DiceLoss**

### 评估指标

训练过程中输出：
- `cls_loss` — 分类损失
- `dice_loss` — Dice 损失
- `total_loss` — 总损失
- `mAP@50` — IoU 阈值 0.5 的 mAP
- `mAP@50-95` — IoU 阈值 0.5~0.95（步长 0.05）的平均 mAP

训练结束后绘制 P-R 曲线、Precision 曲线、Recall 曲线、F1 曲线。

---

## 推理脚本

### `infer_demo.py` — 推理演示
- 加载 `.pth.tar` 权重，对目录下图像做分割预测
- 输出热力图 + 原图拼接结果
- 可用于快速验证模型效果

### `infer_single.py` — 单张图像推理
- 对 `dataset/valid` 中的单张图像推理
- 输出：**原图 | 预测**（绿色叠加 + 绿色轮廓）对比图
- 保存为 `*_final_result.jpg`

### `infer_onnx.py` — ONNX 推理（无 PyTorch 依赖）
- 使用 ONNX Runtime 推理，适合 Orange Pi Zero 3 等 ARM 边缘设备
- 支持命令行参数：
  ```bash
  python infer_onnx.py test.jpg --model dataset1_scratch_sim.onnx --size 256 256
  ```
- 输出：**原图 | 预测叠加**（绿色半透明 + 绿色轮廓）对比图

### `infer_dataset1.py` — dataset1/test 批量推理
- 使用 `dataset1_scratch` 模型，对 dataset1/test 所有图像推理
- 输出：**原图 | 预测叠加（绿色）| 掩膜** 三栏拼接图
- 图像 resize 到 256×256

### `infer_dataset1_valid.py` — dataset1/valid 推理 + 真值对比
- 推理 + 加载 YOLO 格式真值标注
- 输出：**原图 | 预测（绿色）| 真值（蓝色）** 三栏对比图
- 同时生成 HTML 文件打包所有结果

### `infer_dataset1_pack.py` — dataset1 推理 + HTML 打包
- 推理 + 加载 YOLO 真值
- 可视化：原图 | 预测（绿色轮廓+叠加）| 真值（红色/蓝色轮廓）
- 生成 HTML 页面，包含所有图像的对比展示

### `infer_dataset3.py` — dataset3/test 批量推理
- 使用 `chula` 模型（512×640），对 dataset3/test 推理
- 输出三栏拼接：**原图 | 预测叠加（绿色）| 掩膜**

### `infer_dataset3_valid.py` — dataset3/valid 推理 + COCO 真值对比
- 推理 + 加载 COCO 格式真值标注
- 输出：**原图 | 预测（绿色）| 真值（蓝色）** 三栏对比图
- 同时生成 HTML 文件打包

---

## ONNX 导出 (`export_onnx.py`)

将训练好的 `.pth.tar` 权重导出为 ONNX 格式，适用于边缘设备部署：

```bash
python export_onnx.py
```

- 支持动态 batch
- 自动验证 ONNX 模型正确性
- 输出 `.onnx` 文件到 `runs/<model_name>/`

---

## U-Lite + Snake 流水线 (`ulite_snake_pipeline.py`)

两阶段分割方法：

### Stage 1: U-Lite 粗分割
- 模型推理得到 heatmap → 动态阈值二值化 → 形态学去噪 → 取最大连通域

### Stage 2: Snake 精修
- 以 U-Lite 轮廓的最小外接圆（外扩 10%）初始化 Snake
- Active Contour 模型在平滑灰度图上迭代收敛，贴合真实边缘
- 无检测结果时以全局阈值最大区域兜底

### 输出
- **黄色** = Snake 精修轮廓
- **蓝色** = U-Lite 原始轮廓

---

## 数据预处理 (`fileprocessing.py`)

将 ISAT / LabelMe JSON 格式标注转换为 YOLO 格式 txt：

| 函数 | 说明 |
|---|---|
| `getData230719()` | 从 ISAT/LabelMe JSON 提取标注，转换为 YOLO 格式 |
| `getData230725()` | 按边界框裁剪目标生成小图（带 padding） |
| `getCropChula230725()` | 从 chula 数据集按目标裁剪小图，生成 crop_chula |
| `getChula230728()` | 将 Dataset 目录下的 JSON 标注转码为 YOLO 格式 |
| `splitTrainValid()` | 按 9:1 划分训练/验证集，生成 txt 划分文件 |

---

## 数据集 (`data/`)

### 数据集说明

| 数据集 | 图像尺寸 | 目标数 | 类别 | 用途 |
|---|---|---|---|---|
| `chula/` | ~512×640 | 多目标 | hollow, solid | RBC 医学图像分割 |
| `crop_chula/` | ~64×64 | 单目标 | hollow, solid | 单目标快速分割 |
| `dataset1_ulite/` | 256×256 | 单目标 | 自定义 | 通用分割训练 |
| `dataset3_ulite/` | 512×640 | 单目标 | 自定义 | 大图分割训练 |

### 数据格式要求

```
- data_directory/
  ├── images/          # 图像文件（.jpg / .png）
  ├── labels/          # YOLO 格式标注（.txt）
  │                    # 每行：class_id x1 y1 x2 y2 ... xn yn
  ├── classes.txt      # 类别名（每行一个）
  ├── train.txt        # 训练集文件名（不带后缀）
  ├── valid.txt        # 验证集文件名（不带后缀）
  └── test.txt         # 测试集文件名（不带后缀）
```

### YAML 配置（`param.yaml`）

包含三部分：
1. **模型配置** — `in_channels`、`num_classes`、`channels`、`act`、`use_my`
2. **数据增强** — HSV 抖动、几何变换、Mosaic、Copy-Paste 等（参考 YOLOv5）
3. **训练参数** — `lr0`、`momentum`、`weight_decay`

---

## 工具模块 (`utils/`)

### `utils/__init__.py`
导出所有常用工具函数和类。

### `utils/tools.py`

| 函数 | 说明 |
|---|---|
| `check_savedir(path)` | 检查/创建保存目录，已存在时询问是否覆盖 |
| `intersect_dicts(da, db)` | 求两字典的键值交集，用于加载预训练权重时只加载匹配层 |
| `load_model(path, device, cfg)` | 从文件加载模型权重 |
| `save_checkpoint(state, is_best, ...)` | 保存检查点，保留最优和最新 |
| `plot_results(log_file, save_file)` | 绘制训练过程的 loss / mAP 曲线 |
| `plot_pr_curve(...)` | 绘制 P-R 曲线 |
| `plot_mc_curve(...)` | 绘制多类别指标随阈值变化曲线 |

### `utils/losses.py`

| 函数/类 | 说明 |
|---|---|
| `dice_coeff(input, target)` | 计算 Dice 系数 |
| `multiclass_dice_coeff(input, target)` | 多类别平均 Dice 系数 |
| `dice_loss(input, target)` | Dice Loss = 1 - Dice Coefficient |
| `DiceLoss` | Dice Loss 的 PyTorch Module 封装 |

### `utils/metrics.py`

| 函数 | 说明 |
|---|---|
| `calc_mask_iou(pred, target)` | 计算掩膜 IoU |
| `calc_map(pred, target)` | 计算 mAP@50 和 mAP@50-95 |
| `calc_prf(pred, target)` | 计算不同阈值下的 Precision、Recall、F1 |
| `calc_metrics_item(pred, target)` | 批量计算评估所需统计数据 |

### `utils/yolo_datasets.py`

核心数据加载模块：
- `YoloData` — 继承 `torch.utils.data.Dataset`，支持 YOLO 多边形标注解析
- `load_image` — 延迟加载图像、解析轮廓、计算中心点
- 数据增强（参考 YOLOv5）：
  - HSV 色域抖动（hue, saturation, value）
  - 几何变换（旋转、平移、缩放、裁剪、翻转）
  - Mosaic 拼接
  - Copy-Paste 粘贴增强
- `get_yolotrain_loader()` / `get_yoloval_loader()` — 获取训练/验证 DataLoader
- 支持 RAM 缓存和磁盘缓存

### `utils/mask_datasets.py`
- 掩膜数据集框架（**未实现**，仅定义了 `MaskData` 类骨架）

---

## 训练输出 (`runs/dataset1_scratch/`)

| 文件 | 说明 |
|---|---|
| `args.yaml` | 训练命令行参数存档 |
| `cfg.yaml` | 模型配置与训练超参数 |
| `best.pth.tar` | 验证集 loss 最优的模型权重 |
| `last.pth.tar` | 最后一个 epoch 的模型权重 |
| `result.csv` | 训练过程日志（每行：train_cls, train_dice, train_total, train_mAP50, train_mAP50-95, valid_cls, valid_dice, valid_total, valid_mAP50, valid_mAP50-95） |
| `result.png` | loss / mAP 训练曲线图 |
| `dataset1_scratch.onnx` | 导出的 ONNX 模型 |
| `dataset1_scratch_sim.onnx` | 简化版 ONNX 模型 |
| `PR_curve.png` | Precision-Recall 曲线 |
| `P_curve.png` | Precision-阈值 曲线 |
| `R_curve.png` | Recall-阈值 曲线 |
| `F1_curve.png` | F1 Score-阈值 曲线 |

### 训练结果（dataset1_scratch, 300 epochs）

| 指标 | 初值（Epoch 1） | 终值（Epoch 300） |
|---|---|---|
| train_loss | 1.562 | 0.087 |
| valid_loss | 1.559 | 0.072 |
| train_mAP@50 | 0.067 | 0.905 |
| valid_mAP@50 | 0.005 | 0.909 |
| valid_mAP@50-95 | 0.001 | 0.907 |

---

## 其他

### `test_datasets.py` — 数据集调试
- `multiDemo()` — 测试多目标数据集加载与数据增强效果
- `singleDemo()` — 测试单目标数据集加载与数据增强效果
- 保存增强后的图像和掩膜到临时目录，用于可视化验证

### `requirements.txt` — 依赖

```
torch>=1.7.0
torchvision>=0.8.1
numpy>=1.18.5
matplotlib>=3.2.2
opencv-python>=4.6.0
tqdm>=4.64.0
PyYAML>=5.3.1
Pillow>=7.1.2
```

### 可视化颜色约定

| 颜色 | BGR 值 | 含义 |
|---|---|---|
| 绿色 | `(0, 255, 0)` | **预测掩膜**（半透明叠加）/ 预测轮廓 |
| 蓝色 | `(255, 0, 0)` | **真值轮廓**或 U-Lite 原始轮廓 |
| 黄色 | `(0, 255, 255)` | **Snake** 精修轮廓 |
| 红色 | `(0, 0, 255)` | 真值轮廓（部分脚本用） |

---

## 引用

- 论文：[1M parameters are enough? A lightweight CNN-Based model for medical image segmentation](https://mp.weixin.qq.com/s/tzITXfpMpgQaUEQNvQKzdQ)
- 原始实现：[duong-db/U-Lite](https://github.com/duong-db/U-Lite)
- 本仓库：[xujinhui-coding/ulite-bladder-volume-system](https://github.com/xujinhui-coding/ulite-bladder-volume-system)
