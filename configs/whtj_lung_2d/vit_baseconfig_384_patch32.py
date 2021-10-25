# dataset settings
dataset_type = 'GeneralMedicalDataset2D'
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropWithAnnotation'),
    dict(type='Resize', size=(384, 384)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropWithAnnotation'),
    dict(type='Resize', size=(384, 384)),
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
        type='VisionTransformer',
        num_layers=12,
        embed_dim=768,
        num_heads=12,
        img_size=384,
        patch_size=32,
        in_channels=3,
        feedforward_channels=3072,
        drop_rate=0.1),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=num_classes,
        in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[300, 500, 650])
runner = dict(type='EpochBasedRunner', max_epochs=800)

log_config = dict(
    interval=2,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
checkpoint_config = dict(by_epoch=True, interval=2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/pretrain/' \
            'vit_base_patch32_384.pth'
resume_from = None
workflow = [('train', 1)]
