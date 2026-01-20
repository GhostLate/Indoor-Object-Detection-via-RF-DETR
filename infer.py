import argparse
import os

from rfdetr import RFDETRMedium
from tqdm import tqdm

from core.data import CocoDataLoader
from core.utils import annotate_image
from core.visualization import draw_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RF-DETR predictions on images.")
    parser.add_argument("--weights", type=str, help="Path to pre-trained RF-DETR model weights.", default='./output/epoch-10/checkpoint_best_total.pth')
    parser.add_argument("--data-dir", type=str, help="Directory containing coco dataset.", default='./dataset/')
    parser.add_argument("--save-dir", type=str, help="Directory to save annotated images.", default='./output/test-10')
    parser.add_argument("--split", type=str, help="COCO Split: train | test | valid", default='test')
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for predictions.")
    parser.add_argument("--save-imgs", type=bool, default=True, help="Save images")
    return parser.parse_args()


def main():
    args = parse_args()

    config = dict(
        pretrain_weights=args.weights,
        batch_size=12,
        num_classes=7,
    )
    data_loader = CocoDataLoader(args.data_dir, args.split)
    model = RFDETRMedium(**config)

    # Process each image
    for sample in tqdm(data_loader):
        detections = model.predict(sample['image'], threshold=args.confidence)

        pred_image = annotate_image(sample['image'], detections, data_loader.class_names)
        gt_image = annotate_image(sample['image'], sample['annotations'], data_loader.class_names)

        if args.save_imgs:
            draw_plots({'pred': pred_image, 'gt': gt_image}, show=False,
                       save_path=os.path.join(args.save_dir, sample['image_meta']["file_name"]))
        else:
            draw_plots({'pred': pred_image, 'gt': gt_image}, show=True)

    print(f"All images was saved at: {args.save_dir}")

if __name__ == "__main__":
    main()
