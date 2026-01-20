import supervision as sv
from PIL import Image
import numpy as np


def annotate_image(image: Image.Image, detections: sv.Detections, class_names):
    annotated_image = np.array(image)
    annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
    labels = [f"{class_names[pred[3]]} {pred[2]:.2f}" if pred[2] is not None else class_names[pred[3]] for pred in detections]
    annotated_image = sv.LabelAnnotator(smart_position=True).annotate(annotated_image, detections, labels)
    return annotated_image
