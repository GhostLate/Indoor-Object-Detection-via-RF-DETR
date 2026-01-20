import json
import os
from collections import defaultdict
from typing import Iterator

import supervision as sv
from PIL import Image
import numpy as np


class CocoDataLoader:
    def __init__(self, data_dir: str, split: str):
        self.data_dir = data_dir
        self.split = split

        with open(os.path.join(self.data_dir, self.split, "_annotations.coco.json"), "r") as f:
            data = json.load(f)

        self.class_names = {c["id"]: c["name"] for c in data["categories"]}
        self.images = data["images"]

        self.annotations = defaultdict(list)
        for annot in data["annotations"]:
            annot['class_name'] = self.class_names[annot["category_id"]]
            self.annotations[annot['image_id']].append(annot)

        self.sv_annotations = {}
        for img_id, annotations in self.annotations.items():
            xyxy = np.array([annot['bbox'] for annot in annotations])
            xyxy[:, 2:] += xyxy[:, :2]
            confidence = np.ones(xyxy.shape[0])
            class_id = np.array([annot['category_id'] for annot in annotations])

            detections = sv.Detections(
                xyxy=xyxy,
                confidence=None,
                class_id=class_id,
            )
            self.sv_annotations[img_id] = detections

    def __iter__(self) -> Iterator[dict]:
        for image_meta in self.images:
            image = Image.open(os.path.join(self.data_dir, self.split, image_meta['file_name']))
            if image_meta['id'] in self.sv_annotations:
                yield dict(image=image, annotations=self.sv_annotations[image_meta['id']], image_meta=image_meta)

    def __len__(self):
        return len(self.images)
