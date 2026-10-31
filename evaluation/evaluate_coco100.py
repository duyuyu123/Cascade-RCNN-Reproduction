import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

root = Path("data/coco")
pred_dir = Path("outputs/coco100/preds")
eval_dir = Path("evaluation")
eval_dir.mkdir(exist_ok=True)

ann_file = root / "annotations/instances_val2017_100.json"

# 读取100张图片的真实标注
coco_gt = COCO(str(ann_file))

image_ids = sorted(coco_gt.getImgIds())
valid_ids = set(image_ids)

print("验证图片数量:", len(image_ids))

# 读取逐张预测JSON，转换成COCO格式
results = []

for image_id in image_ids:
    filename = f"{image_id:012d}.json"
    pred_file = pred_dir / filename

    if not pred_file.exists():
        raise FileNotFoundError(f"缺少预测文件: {pred_file}")

    with open(pred_file, "r") as f:
        pred = json.load(f)

    for label, score, bbox in zip(
        pred["labels"], pred["scores"], pred["bboxes"]
    ):
        # MMDetection标签从0开始，COCO category_id按标注文件映射
        category_id = coco_gt.getCatIds()[int(label)]

        x1, y1, x2, y2 = bbox

        results.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1)
            ],
            "score": float(score)
        })

# 保存COCO格式预测结果
pred_json = eval_dir / "predictions_coco100.json"

with open(pred_json, "w") as f:
    json.dump(results, f)

print("有效图片:", len(image_ids))
print("参与评估的预测框:", len(results))

# COCO官方评估
coco_dt = coco_gt.loadRes(str(pred_json))
evaluator = COCOeval(coco_gt, coco_dt, "bbox")

evaluator.params.imgIds = image_ids
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()

# 保存结果
names = [
    "AP", "AP50", "AP75",
    "AP_small", "AP_medium", "AP_large"
]

values = evaluator.stats[:6]

result_file = eval_dir / "evaluation_results.txt"

with open(result_file, "w") as f:
    f.write("Cascade R-CNN COCO 100 Images Evaluation\n")
    f.write("=" * 45 + "\n")
    f.write("Images: 100\n")
    f.write("Ground truth annotations: 719\n")
    f.write(f"Predictions: {len(results)}\n\n")

    for name, value in zip(names, values):
        f.write(f"{name}: {value:.4f} ({value * 100:.2f})\n")

print("\n评估结果已保存:", result_file)
