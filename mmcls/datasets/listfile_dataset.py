import os

from . import BaseDataset
from .import DATASETS
from .pipelines import Compose
import numpy as np
from mmcls.models.losses import accuracy
from mmcls.core.evaluation import precision_recall_f1, support, auc
@DATASETS.register_module()
class ListFileDataset(BaseDataset):
    def __init__(self, data_prefix, pipeline, classes=None, ann_file=None, test_mode=False,
                 split_char=' '):
        self.data_prefix = data_prefix
        self.pipeline = Compose(pipeline)
        self.classes = classes
        if ann_file is None:
            raise ValueError(f"Must Specify annotation file in "
                             f"{self.__class__.__name__}")
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.split_char = split_char
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []
        lines = open(self.ann_file).readlines()
        for line in lines:
            file_name, label = line.strip().split(self.split_char)
            data_info = dict(img_prefix=self.data_prefix)
            data_info['img_info'] = {'filename': file_name}
            data_info['gt_label'] = np.array(int(label))
            data_infos.append(data_info)
        return data_infos
    def __len__(self):
        return len(self.data_infos)

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support', 'auc'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metirc {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        if 'auc' in metrics:
            auc_score = auc(results, gt_labels)
            eval_results['auc'] = auc_score

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values
        # eval_results['gt_labels'] = gt_labels.tolist()
        return eval_results