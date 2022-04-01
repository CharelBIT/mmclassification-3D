# dataset settings
dataset_type = 'Hie_Dataset'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type='ExtractDataFromObj'),
    dict(type='NormalizeMedical', norm_type='wcww',
         window_center=30,
         window_width=60),
    dict(type='ResizeMedical', size=(512, 512, 24)),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='ConcatImage'),
    # dict(type='ImageToTensor', keys=['img']),

    dict(type='ToTensor', keys=['gt_label', 'img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type='ExtractDataFromObj'),
    dict(type='NormalizeMedical', norm_type='wcww',
         window_center=30,
         window_width=60),
    dict(type='ResizeMedical', size=(512, 512, 24)),
    dict(type='ToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix='/gruntdata1/charelchen.cj/workDir/dataset/whtj_stroke/CT_nifty_data',
        ann_file='/gruntdata1/charelchen.cj/workDir/dataset/whtj_stroke/train_0.txt',
        pipeline=train_pipeline,
        modes=['image']),
    val=dict(
        type=dataset_type,
        data_prefix='/gruntdata1/charelchen.cj/workDir/dataset/whtj_stroke/CT_nifty_data',
        ann_file='/gruntdata1/charelchen.cj/workDir/dataset/whtj_stroke/test_0.txt',
        pipeline=test_pipeline,
        modes=['image']))
evaluation = dict(interval=2, metric=['accuracy', 'precision', 'recall', 'f1_score', 'support', 'auc'])


norm_cfg = dict(type='BN3d', requires_grad=True)
conv_cfg = dict(type='Conv3d')
num_classes = 2
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetV1dHIE',
        depth=18,
        in_channels=1,
        in_dims=3,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        init_cfg=[
             dict(type='Kaiming', layer=['Conv3d']),
             dict(
                 type='Constant',
                 val=1,
                 layer=['_BatchNorm', 'GroupNorm', 'BN3d'])
         ]
    ),
    neck=dict(type='GlobalAveragePooling', dim=3),
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
lr_config = dict(policy='step', step=[40, 80, 120])
runner = dict(type='EpochBasedRunner', max_epochs=160)

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
load_from = None
resume_from = None
work_dir = 'work_dir/whtj_stroke_resnetV1d18_baseconfig_pretraineds'
workflow = [('train', 1)]
