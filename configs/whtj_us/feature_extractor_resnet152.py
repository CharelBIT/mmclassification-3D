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
    dict(type='Resize', size=(256, -1)),
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
        img_dir='/gruntdata/workDir/dataset/whtj_us/coco/noenc_adj/image',
        ann_file='/gruntdata/workDir/dataset/whtj_us/coco/noenc_adj/annotation/instance_all.json',
        split='/gruntdata/workDir/dataset/whtj_us/coco/noenc_adj/total_datalist.txt',
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
        type='ResNet',
        depth=152,
        num_stages=4,
        out_indices=(3, 2, 1, 0),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))