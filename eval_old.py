import json
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from rfdetr import RFDETRMedium
from rfdetr.main import populate_args
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate
from rfdetr.util.misc import collate_fn
from rfdetr.models import build_criterion_and_postprocessors


def visualize_metrics(class_map):
    metrics = ['map@50:95', 'map@50', 'precision', 'recall']
    df = pd.DataFrame(class_map)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # Flatten 2D array of axes to 1D for easy iteration

    sns.set_theme(style="whitegrid")

    for i, metric in enumerate(metrics):
        ax = axes[i]

        sns.barplot(
            data=df,
            x='class',
            y=metric,
            ax=ax,
            palette='viridis',
            edgecolor='black'
        )
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)  # Uniform scale for easy comparison
        ax.set_ylabel("Score")
        ax.set_xlabel("")  # Hide x-label to reduce clutter

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)

    plt.tight_layout()
    plt.show()


def get_class_names(args: dict):
    if args['dataset_file'] == "roboflow":
        with open(
                os.path.join(args['dataset_dir'], "train", "_annotations.coco.json"), "r"
        ) as f:
            anns = json.load(f)
            class_names = [c["name"] for c in anns["categories"]]
            return class_names
    else:
        return []


def get_all_kwargs(model, **kwargs):
    config = model.get_train_config(**kwargs)

    train_config = config.dict()
    model_config = model.model_config.dict()
    model_config.pop("num_classes")
    if "class_names" in model_config:
        model_config.pop("class_names")

    if "class_names" in train_config and train_config["class_names"] is None:
        train_config["class_names"] = kwargs['class_names']

    for k, v in train_config.items():
        if k in model_config:
            model_config.pop(k)
        if k in config:
            config.pop(k)

    all_kwargs = {**model_config, **train_config, **kwargs}
    return all_kwargs


def evaluate_dataset(config, split: str):
    config['class_names'] = get_class_names(config)

    model = RFDETRMedium(**config)

    all_kwargs = get_all_kwargs(model, **config)
    args = populate_args(**all_kwargs)

    dataset = build_dataset(image_set=split, args=args, resolution=args.resolution)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = DataLoader(dataset, args.batch_size, sampler=sampler,
                                  drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset)
    criterion, postprocess = build_criterion_and_postprocessors(args)

    stats, coco_evaluator = evaluate(
        model.model.model, criterion, postprocess, data_loader, base_ds, args.device, args)

    return stats, coco_evaluator


if __name__ == '__main__':
    config = dict(
        pretrain_weights='./output/epoch-10/checkpoint_best_total.pth',
        dataset_dir='./dataset',
        batch_size=12,
        output_dir='./output_tmp',
        dataset_file="roboflow",
        num_classes=7,
    )
    stats, coco_evaluator = evaluate_dataset(config, 'test')

    visualize_metrics(stats['results_json']['class_map'])
