import argparse
import os
import json

import natsort
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def load_coco_stats(data_dir):
    """Parses COCO JSONs and aggregates stats."""
    seq_data = []
    bbox_data = []

    dirs = natsort.natsorted(os.listdir(data_dir))
    for seq_name in tqdm(dirs):
        file_path = os.path.join(data_dir, seq_name, "_annotations.coco.json")

        if not os.path.isfile(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        images = data.get('images', [])
        annotations = data.get('annotations', [])
        categories = data.get('categories', [])

        n_images = len(images)
        n_annots = len(annotations)
        n_cats = len(categories)

        cat_map = {cat['id']: cat['name'] for cat in categories}

        seq_data.append({
            'Sequence': seq_name,
            'Images': n_images,
            'Annotations': n_annots,
            'Categories': n_cats,
            'Avg Objects/Img': n_annots / n_images if n_images > 0 else 0
        })

        for ann in annotations:
            cat_name = cat_map.get(ann['category_id'], 'Unknown')
            width = ann['bbox'][2]
            height = ann['bbox'][3]
            area = width * height

            bbox_data.append({
                'Sequence': seq_name,
                'Category': cat_name,
                'Area': area,
                'AspectRatio': width / height if height > 0 else 0
            })
    return pd.DataFrame(seq_data), pd.DataFrame(bbox_data)


def visualize(data_dir):
    df_seq, df_bbox = load_coco_stats(data_dir)

    if df_seq.empty:
        print("No data found. Check your paths.")
        return

    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2)

    # 1. Volume Overview (Images vs Annotations)
    ax1 = fig.add_subplot(gs[0, 0])
    df_melted = df_seq.melt(id_vars="Sequence", value_vars=["Images", "Annotations"], var_name="Metric",
                            value_name="Count")
    sns.barplot(data=df_melted, x="Sequence", y="Count", hue="Metric", ax=ax1, palette="viridis")
    ax1.set_title("Dataset Volume per Sequence", fontsize=14, fontweight='bold')

    # 2. Annotation Density
    ax2 = fig.add_subplot(gs[0, 1])
    sns.lineplot(data=df_seq, x="Sequence", y="Avg Objects/Img", marker="o", ax=ax2, color="crimson", linewidth=2.5)
    ax2.set_title("Object Density (Avg Annotations per Image)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Count")

    # 3. Class Distribution Heatmap
    ax3 = fig.add_subplot(gs[1, 0])

    class_counts = df_bbox.groupby(['Sequence', 'Category']).size().unstack(fill_value=0)
    sns.heatmap(class_counts, annot=True, fmt="d", cmap="YlGnBu", ax=ax3)
    ax3.set_title("Class Balance Heatmap", fontsize=14, fontweight='bold')

    # 4. Bounding Box Area Distribution (Boxplot)
    ax4 = fig.add_subplot(gs[1, 1])

    sns.boxplot(data=df_bbox, x="Sequence", y="Area", ax=ax4, palette="pastel", showfliers=False)
    ax4.set_title("Bounding Box Area Distribution (Outliers Hidden)", fontsize=14, fontweight='bold')
    ax4.set_ylabel("Pixel Area ($w \\times h$)")

    plt.tight_layout()
    plt.show()

    # --- Print Text Summary ---
    print("\n--- Summary Statistics ---")
    print(df_seq.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RF-DETR predictions on images.")
    parser.add_argument("--data-dir", type=str, help="Directory containing dataset.", default='./dataset/')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    visualize(data_dir=args.data_dir)
