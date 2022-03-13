dataset_type = 'ContinousSliceDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropWithAnnotation'),
    dict(
        type='Resize',
        size=(256, -1),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        img_dir='/gruntdata/workDir/dataset/whtj_us/coco/enc_adj/image',
        ann_file='/gruntdata/workDir/dataset/whtj_us/coco/enc_adj/annotation/instance_all.json',
        split='/gruntdata/workDir/dataset/whtj_us/coco/enc_adj/total_datalist.txt',
        pipeline=train_pipeline,
        ),
    # val=dict(
    #     type=dataset_type,
    #     img_dir='/gruntdata/workDir/dataset/whtj_us/coco/recurrence/CEUS/image',
    #     ann_file='/gruntdata/workDir/dataset/whtj_us/coco/recurrence/CEUS/annotation/instance_all.json',
    #     split='/gruntdata/workDir/dataset/whtj_us/recurrence/val_0_123.txt',
    #     pipeline=test_pipeline,
    #     ),
    # test=dict(
    #     # replace `data/val` with `data/test` for standard test
    #     type=dataset_type,
    #     img_dir='/gruntdata/workDir/dataset/whtj_us/coco/recurrence/CEUS/image',
    #     ann_file='/gruntdata/workDir/dataset/whtj_us/coco/recurrence/CEUS/annotation/instance_all.json',
    #     split='/gruntdata/workDir/dataset/whtj_us/recurrence/val_0_123.txt',
    #     pipeline=test_pipeline,
    #     )
)


model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer', arch='small', img_size=224,
        drop_path_rate=0.3),
    neck=dict(type='GlobalAveragePooling', dim=1),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5)
    ]))

