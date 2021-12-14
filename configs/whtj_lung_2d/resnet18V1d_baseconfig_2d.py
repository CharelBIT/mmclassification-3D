# dataset settings
dataset_type = 'GeneralMedicalDataset2D'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropWithAnnotation'),
    dict(type='Resize', size=(224, 224)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropWithAnnotation'),
    dict(type='Resize', size=(224, 224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/2dSlice_clip_norm/image',
        ann_file='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/2dSlice_clip_norm/annotation/instance_all.json',
        split='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/train_split_130.txt',
        pipeline=train_pipeline,
        ),
    val=dict(
        type=dataset_type,
        img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/2dSlice_clip_norm/image',
        ann_file='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/2dSlice_clip_norm/annotation/instance_all.json',
        split='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/val_split_130.txt',
        pipeline=test_pipeline,
        ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/2dSlice_clip_norm/image',
        ann_file='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/2dSlice_clip_norm/annotation/instance_all.json',
        ssplit='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/test_split_130.txt',
        pipeline=test_pipeline,
        ))
evaluation = dict(interval=2, metric=['accuracy', 'precision', 'recall', 'f1_score', 'support', 'auc'])


# norm_cfg = dict(type='BN3d', requires_grad=True)
# conv_cfg = dict(type='Conv3d')
num_classes = 2
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetV1d',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[50, 100, 150])
runner = dict(type='EpochBasedRunner', max_epochs=200)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
checkpoint_config = dict(by_epoch=True, interval=2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/pretrain/' \
            'resnet18_batch256_imagenet_20200708-34ab8f90.pth'
resume_from = None
workflow = [('train', 1)]
