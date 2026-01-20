import argparse
import os

import fiftyone as fo


def main(data_dir: str, split: str):
    coco_dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=os.path.join(data_dir, split),
        labels_path=os.path.join(data_dir, split, '_annotations.coco.json')
    )
    session = fo.launch_app(coco_dataset)
    session.wait()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RF-DETR predictions on images.")
    parser.add_argument("--data-dir", type=str, help="Directory containing dataset.", default='./dataset/')
    parser.add_argument("--split", type=str, help="Dataset split to load", default='test')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.split)
