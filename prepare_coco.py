import argparse
import json
import os
import xml.etree.ElementTree as ET

from PIL import Image


def convert2coco(idx: int, dataset_dir: str):
    tree = ET.parse(os.path.join(dataset_dir, f"./annotation/annotation_s{idx}.xml"))
    root = tree.getroot()

    coco_output = {
            "info": {
                "description": root.find('name').text.strip() if root.find('name') is not None else "",
                "year": 2018,
                "contributor": root.find('comment').text.strip() if root.find('comment') is not None else "",
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }

    category_map = {}  # "label_name": category_id
    category_id_counter = 0
    annotation_id_counter = 0
    image_id_counter = 0

    images_tag = root.find('images')

    for image_elem in images_tag.findall('image'):
        filename = image_elem.get('file')

        img_w, img_h = 0, 0
        img_path = os.path.join(dataset_dir, f"./sequence_{idx}", filename)
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except FileNotFoundError:
            print(f"Warning: Image {filename} not found. Width/Height set to 0.")

        image_info = {
            "id": image_id_counter,
            "file_name": filename,
            "width": img_w,
            "height": img_h
        }
        coco_output["images"].append(image_info)

        for box in image_elem.findall('box'):
            top = int(box.get('top'))
            left = int(box.get('left'))
            width = int(box.get('width'))
            height = int(box.get('height'))

            label_name = box.find('label').text

            # Manage Categories
            if label_name not in category_map:
                category_map[label_name] = category_id_counter
                coco_output["categories"].append({
                    "id": category_id_counter,
                    "name": label_name,
                    "supercategory": "none"
                })
                category_id_counter += 1

            cat_id = category_map[label_name]

            # Add Annotation Entry
            ann_info = {
                "id": annotation_id_counter,
                "image_id": image_id_counter,
                "category_id": cat_id,
                "bbox": [left, top, width, height],  # COCO bbox format
                "area": width * height,
                "iscrowd": 0,
                "segmentation": []  # Bbox only, no segmentation
            }
            coco_output["annotations"].append(ann_info)
            annotation_id_counter += 1

        image_id_counter += 1

    output_json = os.path.join(dataset_dir, f"./sequence_{idx}", "_annotations.coco.json")
    with open(output_json, 'w') as f:
        json.dump(coco_output, f, indent=4)

    print(f"Conversion complete! Saved to {output_json}")
    print(f"Stats: {len(coco_output['images'])} images, {len(coco_output['annotations'])} annotations.")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RF-DETR predictions on images.")
    parser.add_argument("--data-dir", type=str, help="Directory containing dataset.", default='./dataset/')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for i in range(1,7):
        convert2coco(i, args.data_dir)
