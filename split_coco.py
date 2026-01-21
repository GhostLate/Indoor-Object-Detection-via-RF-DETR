import argparse
import json
import os
import shutil

import natsort
import numpy as np


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)  # Add indent=2 for readability if needed, but it increases file size


def initialize_coco():
    return {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }


def split_list_by_ratios(data: list[dict], split_ratios: dict):
    ratios = list(split_ratios['splits'].values())
    names = list(split_ratios['splits'].keys())

    max_seq = split_ratios['max_seq']
    assert all([ratio * max_seq >= 1 for ratio in ratios]), 'small max_seq in split_ratios'

    seq_ids = [i for i in range(max_seq, len(data), max_seq)]
    seqs = np.split(data, seq_ids[:-1])

    splits = {name: [] for name in names}
    for seq in seqs:
        split_indices = (np.cumsum(ratios) * len(seq)).astype(int)
        seq_splits = np.split(seq, split_indices[:-1])
        for i, seq_split in enumerate(seq_splits):
            splits[names[i]] += seq_split.tolist()
    return splits


def split_list_by_ratios2(data: list[dict], split_ratios: dict):
    ratios = list(split_ratios['splits'].values())
    names = list(split_ratios['splits'].keys())

    split_indices = (np.cumsum(ratios) * len(data)).astype(int)
    seq_splits = np.split(data, split_indices[:-1])
    splits = {name: seq_splits[i].tolist() for i, name in enumerate(names)}
    return splits


def process_merge(strategy, data_dir, save_dir):
    splits_names = set([split for split in strategy.values() if isinstance(split, str)])
    splits_names.update(set([key for split in strategy.values() if isinstance(split, dict) for key in split['splits'].keys()]))

    for splits_name in splits_names:

        shutil.rmtree(os.path.join(save_dir, splits_name), ignore_errors=True)
        os.makedirs(os.path.join(save_dir, splits_name), exist_ok=True)

    # Prepare output containers
    splits = {splits_name: initialize_coco() for splits_name in splits_names}
    global_categories = {}

    for seq_name, assignment in strategy.items():
        json_path = os.path.join(data_dir, seq_name, "_annotations.coco.json")

        if not os.path.exists(json_path):
            print(f"Skipping {seq_name} (File not found)")
            continue

        print(f"Processing {seq_name}...")
        data = load_json(json_path)

        # Set global categories
        for category in data['categories']:
            if category['name'] not in global_categories:
                global_categories[category['name']] = {
                    'id': len(global_categories),
                    'name': category['name'],
                    'supercategory': category['supercategory'],
                }

        categories_map = {}
        for category in data['categories']:
            categories_map[category['id']] = global_categories[category['name']]['id']

        # Sort and split images by file_name to ensure time-based order
        images = natsort.natsorted(data['images'], key=lambda x: x['file_name'])
        if isinstance(assignment, str):
            groups = {assignment: images}
        else:
            assert isinstance(assignment, dict)
            groups = split_list_by_ratios(images, assignment)

        # Perform the merge for this sequence
        for target_split, img_subsets in groups.items():
            curr_dataset = splits[target_split]

            if not curr_dataset['info']:
                curr_dataset['info'] = data['info']

            img_id_map = dict()
            for img_subset in img_subsets:
                img_id_map[img_subset['id']] = len(curr_dataset['images'])
                img_subset['id'] = len(curr_dataset['images'])

                new_file_name = img_subset['file_name'].split('.')
                new_file_name[-2] = f"{new_file_name[-2]}_{img_subset['id']}"
                new_file_name = '.'.join(new_file_name)
                shutil.copy(os.path.join(data_dir, seq_name, img_subset['file_name']),
                            os.path.join(save_dir, target_split, new_file_name))
                img_subset['file_name'] = new_file_name

                curr_dataset['images'].append(img_subset)

            for annot in data['annotations']:
                if annot['image_id'] in img_id_map:
                    new_annot = annot.copy()
                    new_annot['image_id'] = img_id_map[annot['image_id']]
                    new_annot['category_id'] = categories_map[annot['category_id']]
                    new_annot['id'] = len(curr_dataset['annotations'])
                    curr_dataset['annotations'].append(new_annot)

        for split_name, dataset in splits.items():
            dataset['categories'] = list(global_categories.values())

            out_path = os.path.join(save_dir, split_name, "_annotations.coco.json")
            print(
                f"Saving {split_name}: {len(dataset['images'])} images, {len(dataset['annotations'])} annotations")
            save_json(dataset, out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RF-DETR predictions on images.")
    parser.add_argument("--data-dir", type=str, help="Directory containing dataset.", default='./dataset/')
    parser.add_argument("--save-dir", type=str, help="Directory to save dataset.", default='./dataset/')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # --- Configuration ---
    # Define how sequences are distributed
    # "split" means we chop it up by `max_seq` batches by ratio of train, valid, test.

    strategy = {
        "sequence_1": {'splits': {'train': 0.8, 'valid': 0.2}, 'max_seq' : 5}, # sequence_1 for Train & Valid
        "sequence_2": {'splits': {'test': 0.6, 'valid': 0.4}, 'max_seq' : 5}, # sequence_2 only for Test & Valid
        "sequence_3": {'splits': {'train': 0.8, 'test': 0.1, 'valid': 0.1}, 'max_seq' : 40}, # sequence_3 for Train, Valid & Test, to fix class bias
        "sequence_4": {'splits': {'train': 0.8, 'test': 0.1, 'valid': 0.1}, 'max_seq' : 10}, # sequence_4 for Train, Valid & Test, to fix class bias
        "sequence_5": {'splits': {'train': 0.8, 'valid': 0.2}, 'max_seq' : 5}, # sequence_5 only for Train & Valid
        "sequence_6": {'splits': {'train': 0.8, 'valid': 0.2}, 'max_seq' : 5}, # sequence_6 only for Train & Valid
    }
    process_merge(strategy, args.data_dir, args.save_dir)
