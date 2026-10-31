# Cascade R-CNN 目标检测模型复现实验

## 1. 项目简介

本项目基于 MMDetection 框架，对 Cascade R-CNN 目标检测模型进行基础复现。实验使用 ResNet-50 + FPN 配置及 COCO 预训练权重，完成单图推理、结果可视化、COCO val2017 100 张图片子集评估和推理性能测试。

## 2. 实验环境

| 项目                 | 配置                             |
| ------------------ | ------------------------------ |
| 操作系统               | Linux                          |
| Python             | 3.10                           |
| PyTorch            | 2.1.2+cu121                    |
| CUDA（PyTorch 构建版本） | 12.1                           |
| NVIDIA Driver      | 550.144.03                     |
| GPU                | NVIDIA GeForce RTX 3090（24 GB） |
| NumPy              | 1.26.4                         |
| MMCV               | 2.1.0                          |
| MMEngine           | 0.10.7                         |
| MMDetection        | 3.3.0                          |
| Conda 环境           | `dyy_cascade`                  |

激活环境并进入项目目录：

```bash
conda activate dyy_cascade
cd /home/shu1004/dyy/cascade_rcnn
```

## 3. 模型配置与权重

- 模型：Cascade R-CNN
- Backbone：ResNet-50
- Neck：Feature Pyramid Network（FPN）
- 预训练数据集：COCO
- 配置文件：`checkpoints/cascade-rcnn_r50_fpn_1x_coco.py`
- 权重文件：`checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth`
- 权重文件大小：约 265 MiB

## 4. 目录结构

```text
cascade_rcnn/
├── README.md
├── checkpoints/
│   ├── cascade-rcnn_r50_fpn_1x_coco.py
│   └── cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth
├── configs/
├── data/
│   ├── demo.jpg
│   └── coco/
│       ├── annotations/
│       │   ├── instances_val2017.json
│       │   └── instances_val2017_100.json
│       ├── val2017/
│       └── selected_100.txt
├── evaluation/
│   ├── evaluate_coco100.py
│   ├── predictions_coco100.json
│   └── evaluation_results.txt
└── outputs/
    ├── vis/
    └── coco100/
        ├── preds/
        └── vis/
```

推理过程中生成的逐图预测 JSON 和可视化结果位于 `outputs/coco100/preds/`、`outputs/coco100/vis/`。

## 5. 单张图片推理

使用 MMDetection 的 `DetInferencer` 接口加载模型：

```python
from mmdet.apis import DetInferencer

inferencer = DetInferencer(
    model='checkpoints/cascade-rcnn_r50_fpn_1x_coco.py',
    weights='checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth',
    device='cuda:0'
)

inferencer(
    'data/demo.jpg',
    out_dir='outputs',
    pred_score_thr=0.3
)
```

本实验已成功完成单图推理，并生成检测可视化结果。测试图片中识别出了猫、遥控器等目标。

## 6. COCO 100 张图片子集评估

### 6.1 数据说明

从 COCO val2017 中固定选取 100 张图片，随机种子为 42。该子集包含：

| 项目    | 数量   |
| ----- | ----:|
| 验证图片  | 100  |
| 真实标注框 | 719  |
| 类别    | 80   |
| 模型预测框 | 2651 |

评估使用这 100 张图片对应的真实标注，不将额外单独下载的演示图片计入评估。

### 6.2 评估方式

使用 `pycocotools.COCOeval` 对模型预测结果进行 COCO bbox 评估。已执行的评估命令为：

```bash
python evaluation/evaluate_coco100.py
```

评估结果文件：

```text
evaluation/evaluation_results.txt
```

### 6.3 精度结果

| 指标                | 结果     |
| ----------------- | ------:|
| AP（IoU 0.50:0.95） | 47.45% |
| AP50              | 67.12% |
| AP75              | 53.78% |
| AP-small          | 32.36% |
| AP-medium         | 49.87% |
| AP-large          | 65.05% |

其中，AP 为 IoU 0.50 至 0.95（步长 0.05）下的平均精度；AP50 和 AP75 分别表示 IoU 阈值为 0.50 和 0.75 时的平均精度。AP-small、AP-medium 和 AP-large 分别表示不同目标面积范围下的 AP。

### 6.4 结果分析

在本次 100 张图片子集上，Cascade R-CNN 的 AP 为 47.45%，AP50 为 67.12%，AP75 为 53.78%。不同尺度目标的 AP 存在差异：大目标为 65.05%，中目标为 49.87%，小目标为 32.36%。在该子集上，小目标检测相对困难。

**重要说明：** 以上是 COCO val2017 的 100-image subset 结果，不是完整 COCO val2017 验证集的官方指标，不能直接与论文报告的完整验证集 AP 作严格比较。

## 7. 推理性能测试

测试设置：先预热 5 次，再对同一张图片正式推理 20 次；每次计时前后执行 CUDA 同步。

| 指标              | 结果         |
| --------------- | ----------:|
| 平均推理时间          | 39.46 ms   |
| 最快推理时间          | 39.19 ms   |
| 最慢推理时间          | 39.79 ms   |
| 按平均延迟换算的吞吐率     | 25.34 FPS  |
| PyTorch 峰值显存分配量 | 587.46 MiB |

FPS 按 `1000 / 平均推理时间（ms）` 换算。该数值是单图重复推理的估计吞吐率，不是批量推理或视频端到端测试结果。显存数值是 PyTorch 记录的峰值张量分配量，不等于 GPU 的全部显存占用。服务器 GPU 为共享资源，其他进程可能影响测试结果。

## 8. 实验总结

本项目完成了 Cascade R-CNN 环境配置、官方预训练权重加载、单图检测与可视化、COCO 100 张图片子集精度评估，以及推理延迟和显存测试。实验结果可用于说明该模型的基本运行流程和在当前小规模子集上的检测表现。



## 参考文献与资源

1. Cai, Z. and Vasconcelos, N. “Cascade R-CNN: Delving into High Quality Object Detection.” CVPR, 2018.
2. Chen, K. et al. “MMDetection: Open MMLab Detection Toolbox and Benchmark.” arXiv, 2019.
3. Lin, T.-Y. et al. “Microsoft COCO: Common Objects in Context.” ECCV, 2014.
- MMDetection: https://github.com/open-mmlab/mmdetection
- MMDetection 文档: https://mmdetection.readthedocs.io/
- COCO 数据集: https://cocodataset.org/
