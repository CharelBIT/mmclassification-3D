# dataset settings
dataset_type = 'Hie_Dataset'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_root = '/opt/data/private/project/charelchen.cj/workDir/dataset/hie'
train_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type='ExtractDataFromObj'),
    dict(type='NormalizeMedical', norm_type='full_volume_mean',
         instensity_min_val=0.5,
         instensity_max_val=99.5),
    dict(type='RandomCropMedical', crop_size=(144, 144, 16)),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='ConcatImage'),
    # dict(type='ImageToTensor', keys=['img']),

    dict(type='ToTensor', keys=['gt_label', 'img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type='ExtractDataFromObj'),
    dict(type='NormalizeMedical', norm_type='full_volume_mean',
         instensity_min_val=0.5,
         instensity_max_val=99.5),
    dict(type='CentralCropMedical', crop_size=(144, 144, 16)),
    dict(type='ToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        data_prefix='{}/hie_niigz'.format(data_root),
        ann_file='{}/trainlist_newgroup.txt'.format(data_root),
        pipeline=train_pipeline,
        modes=['t1_zw']),
    val=dict(
        type=dataset_type,
        data_prefix='{}/hie_niigz'.format(data_root),
        ann_file='{}/testlist_newgroup.txt'.format(data_root),
        pipeline=test_pipeline,
        modes=['t1_zw']),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='{}/hie_niigz'.format(data_root),
        ann_file='{}/testlist_newgroup.txt'.format(data_root),
        pipeline=test_pipeline,
        modes=['t1_zw']))
evaluation = dict(interval=2, metric=['accuracy', 'precision', 'recall', 'f1_score', 'support', 'auc'])


norm_cfg = dict(type='BN3d', requires_grad=True)
conv_cfg = dict(type='Conv3d')
num_classes = 2
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetV1dHIE',
        depth=50,
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
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ))

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(policy='step', step=[40, 80, 120])
# runner = dict(type='EpochBasedRunner', max_epochs=160)
#
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
optimizer = dict(
    type='AdamW',
    lr=5e-4 * 128 * 8 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    )
optimizer_config = dict(grad_clip=dict(max_norm=5.0))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=0.1,
    warmup_iters=10000,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=5000)
# yapf:enable
checkpoint_config = dict(by_epoch=True, interval=10)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
