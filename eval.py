import argparse
from tqdm import tqdm

from rfdetr import RFDETRMedium

from core.data import CocoDataLoader
from core.metrics import DetectionMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Eval RF-DETR predictions on COCO split.")
    parser.add_argument("--weights", type=str, help="Path to pre-trained RF-DETR model weights.",
                        default='./output/epoch-10/checkpoint_best_total.pth')
    parser.add_argument("--data-dir", type=str, help="Directory containing coco dataset.", default='./dataset/')
    parser.add_argument("--split", type=str, help="COCO Split: train | test | valid", default='test')
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for predictions.")
    return parser.parse_args()


def main():
    args = parse_args()

    config = dict(
        pretrain_weights=args.weights,
        batch_size=12,
        num_classes=7,
    )
    data_loader = CocoDataLoader(args.data_dir, args.split)
    metrics = DetectionMetrics(data_loader.class_names)
    model = RFDETRMedium(**config)

    # Process each image
    for sample in tqdm(data_loader):
        detections = model.predict(sample['image'], threshold=args.confidence)
        metrics.update(detections, sample['annotations'])

    metrics.visualize(title=f'Eval metrics on `{args.split}` dataset split')


if __name__ == "__main__":
    main()
