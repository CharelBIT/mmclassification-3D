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
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_prefix='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_stroke/CT_nifty_data',
        ann_file='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_stroke/train_ct_0.txt',
        pipeline=train_pipeline,
        modes=['image']),
    val=dict(
        type=dataset_type,
        data_prefix='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_stroke/CT_nifty_data',
        ann_file='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_stroke/test_ct_0.txt',
        pipeline=test_pipeline,
        modes=['image']),
    test=[
        dict(
            type=dataset_type,
            data_prefix='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_stroke/CTpredict_ex_CT_extract',
            ann_file='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_stroke/extendlist.txt',
            pipeline=test_pipeline,
            modes=['image'])
    ]
)
evaluation = dict(interval=2, metric=['accuracy', 'precision', 'recall', 'f1_score', 'support', 'auc'])


norm_cfg = dict(type='BN3d', requires_grad=True)
conv_cfg = dict(type='Conv3d')
num_classes = 2
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNeXtV1dHIE',
        depth=34,
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

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 128 * 8 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=500,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=400)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
checkpoint_config = dict(by_epoch=True, interval=1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
work_dir = 'work_dirs/whtj_stroke_resnextV1d34_baseconfig_pretraineds_lr'
workflow = [('train', 1)]
