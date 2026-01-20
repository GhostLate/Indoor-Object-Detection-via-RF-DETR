from collections import defaultdict

import seaborn as sns
import pandas as pd
import supervision as sv
from matplotlib import pyplot as plt
from supervision.metrics import MeanAverageRecall, MeanAveragePrecision, Precision, Recall, F1Score


class DetectionMetrics:
    def __init__(self, class_names: dict):
        self.mar_metric = MeanAverageRecall()
        self.map_metric = MeanAveragePrecision()
        self.precision_metric = Precision()
        self.recall_metric = Recall()
        self.f1_metric = F1Score()

        self.detections = []
        self.annotations = []
        self.class_names = class_names

    def update(self, detections, annotations):
        self.f1_metric.update(detections, annotations)
        self.recall_metric.update(detections, annotations)
        self.precision_metric.update(detections, annotations)
        self.mar_metric.update(detections, annotations)
        self.map_metric.update(detections, annotations)

        self.detections.append(detections)
        self.annotations.append(annotations)

    def compute(self):
        f1_result = self.f1_metric.compute()
        recall_result = self.recall_metric.compute()
        precision_result = self.precision_metric.compute()
        mar_result = self.mar_metric.compute()
        map_result = self.map_metric.compute()

        result_map = defaultdict(dict)
        # F1 Score
        for class_id in f1_result.matched_classes:
            result_map[self.class_names[class_id]].update(
                {'F1-50': f1_result.f1_per_class[class_id][0], 'F1-50:95': f1_result.f1_per_class[class_id].mean()})
        result_map['all'].update({'F1-50': f1_result.f1_scores[0], 'F1-50:95': f1_result.f1_scores.mean()})

        # Recall Score
        for class_id in recall_result.matched_classes:
            result_map[self.class_names[class_id]].update(
                {'Recall-50': recall_result.recall_per_class[class_id][0],
                 'Recall-50:95': recall_result.recall_per_class[class_id].mean()})
        result_map['all'].update(
            {'Recall-50': recall_result.recall_scores[0], 'Recall-50:95': recall_result.recall_scores.mean()})

        # Precision Score
        for class_id in precision_result.matched_classes:
            result_map[self.class_names[class_id]].update(
                {'Precision-50': precision_result.precision_per_class[class_id][0],
                 'Precision-50:95': precision_result.precision_per_class[class_id].mean()})
        result_map['all'].update({'Precision-50': precision_result.precision_scores[0],
                                  'Precision-50:95': precision_result.precision_scores.mean()})

        # mAR Score
        for class_id in mar_result.matched_classes:
            result_map[self.class_names[class_id]].update(
                {'mAR-50': mar_result.recall_per_class[class_id][0],
                 'mAR-50:95': mar_result.recall_per_class[class_id].mean()})
        result_map['all'].update({'mAR-50': mar_result.recall_per_class.mean(0)[0], 'mAR-50:95': mar_result.recall_per_class.mean(0).mean(0)})

        # mAP Score
        for class_id in map_result.matched_classes:
            result_map[self.class_names[class_id]].update(
                {'mAP-50': map_result.ap_per_class[class_id][0], 'mAP-50:95': map_result.ap_per_class[class_id].mean()})
        result_map['all'].update({'mAP-50': map_result.mAP_scores[0], 'mAP-50:95': map_result.mAP_scores.mean()})

        # Class Confusion Matrix
        conf_mat = sv.ConfusionMatrix.from_detections(
            self.detections, self.annotations, [self.class_names[i] for i in range(len(self.class_names))])

        return result_map, conf_mat

    def visualize(self, title, normalize=True):
        result_map, conf_mat = self.compute()

        x_labels = conf_mat.classes + ['FN']
        y_labels = conf_mat.classes + ['FP']
        data = conf_mat.matrix / (
                    conf_mat.matrix.sum(0).reshape(1, -1) + 1e-8) if normalize else conf_mat.matrix.astype(int)
        mask = data == 0

        result_pd = pd.DataFrame.from_dict(result_map)

        fig = plt.figure(figsize=(25, 15))
        gs = fig.add_gridspec(1, 2)

        ax1 = fig.add_subplot(gs[0])
        sns.heatmap(result_pd, annot=True, fmt='.2', cmap="Blues", square=True,
                    ax=ax1, cbar=False, vmin=0, vmax=1, linewidths=.25)
        ax1.set_title("Class Metrics", fontsize=16, fontweight='bold')
        ax1.set_xlabel("Classes", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Metrics", fontsize=12, fontweight='bold')
        plt.yticks(rotation=0)

        ax2 = fig.add_subplot(gs[1])
        sns.heatmap(data, annot=True, fmt='.2' if normalize else 'd', cmap='YlOrBr', square=True,
                    ax=ax2, cbar=False, xticklabels=x_labels, yticklabels=y_labels, mask=mask)
        ax2.set_title("Class Confusion Matrix", fontsize=16, fontweight='bold')
        plt.yticks(rotation=0)

        fig.suptitle(title, fontsize=20, fontweight='bold')
        plt.show()
