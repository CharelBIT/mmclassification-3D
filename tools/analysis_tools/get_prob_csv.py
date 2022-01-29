import argparse
import os
import warnings
import mmcv
import pandas as pd
import numpy as np
import torch
from mmcv import DictAction
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model

def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default=None, help='output result file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def get_model_prob(model, data_loader):
    results = []
    model.eval()
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
            results.extend(result)
    return results

def netBenefit(y_true, y_prob, threshold):
    y_pred = np.zeros(y_prob.shape)
    y_pred[np.where(y_prob > threshold)] = 1
    num = y_true.size
    tp_num = len(np.intersect1d(np.where(y_true == 1), np.where(y_pred == 1)))
    fp_num = len(np.intersect1d(np.where(y_true == 0), np.where(y_pred == 1)))

    tpr = tp_num / num
    fpr = fp_num / num

    NB = tpr - fpr * threshold / (1 - threshold)
    return NB

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # cfg.data.train['pipeline'] = cfg.data.test['pipeline']
    train_dataset = build_dataset(cfg.data.train)
    test_dataset = build_dataset(cfg.data.test)
    val_dataset = build_dataset(cfg.data.val)
    train_data_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)
    test_data_loader = build_dataloader(
        test_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)

    val_data_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)
    init_model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(init_model)
    checkpoint = load_checkpoint(init_model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES
    if not distributed:
        if args.device == 'cpu':
            model = init_model.cpu()
        else:
            model = MMDataParallel(init_model, device_ids=[0])
        model.CLASSES = CLASSES
        val_results = get_model_prob(model, val_data_loader)
        test_results = get_model_prob(model, test_data_loader)
        train_results = get_model_prob(model, train_data_loader)
    else:
        # val_results = None
        # test_results = None
        raise NotImplementedError
    df = pd.DataFrame()
    for train_result, data_info in zip(train_results, train_dataset.data_infos):
        df.loc[data_info['patient_id'], 'score'] = train_result[1]
        df.loc[data_info['patient_id'], 'train_sample'] = 1
        df.loc[data_info['patient_id'], 'Label'] = int(data_info['gt_label'])
    for val_result, data_info in zip(val_results, val_dataset.data_infos):
        df.loc[data_info['patient_id'], 'score'] = val_result[1]
        df.loc[data_info['patient_id'], 'train_sample'] = 0
        df.loc[data_info['patient_id'], 'Label'] = int(data_info['gt_label'])
    # for test_result, data_info in zip(test_results, test_dataset.data_infos):
    #     df.loc[data_info['patient_id'], 'score'] = test_result[1]
    #     df.loc[data_info['patient_id'], 'train_sample'] = -1
    #     df.loc[data_info['patient_id'], 'Label'] = int(data_info['gt_label'])
    df.to_csv(args.out)


if __name__ == '__main__':
    main()