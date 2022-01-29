# dataset settings
dataset_type = 'GeneralMedicalDataset'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type="LoadAnnotationsFromNIIFile"),
    dict(type='ExtractDataFromObj'),
    dict(type='CropMedicalWithAnnotations'),
    dict(type='NormalizeMedical', norm_type='full_volume_mean',
         instensity_min_val=0.5,
         instensity_max_val=99.5),
    dict(type='ResizeMedical', size=(160, 160, 80)),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='ConcatImage'),
    # dict(type='ImageToTensor', keys=['img']),

    dict(type='ToTensor', keys=['gt_label', 'img']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type="LoadAnnotationsFromNIIFile"),
    dict(type='ExtractDataFromObj'),
    dict(type='CropMedicalWithAnnotations'),
    dict(type='NormalizeMedical', norm_type='full_volume_mean',
         instensity_min_val=0.5,
         instensity_max_val=99.5),
    dict(type='ResizeMedical', size=(160, 160, 80)),
    dict(type='ToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
                    'hie_niigz_mask_seg_0.5x0.5x0.5',
        ann_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
                    'hie_niigz_mask_seg_0.5x0.5x0.5',
        mode='t1_zw',
        split='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/t1_zw_fse_train.txt',
        pipeline=train_pipeline,
        ),
    val=dict(
        type=dataset_type,
        img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
                    'hie_niigz_mask_seg_0.5x0.5x0.5',
        ann_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
                    'hie_niigz_mask_seg_0.5x0.5x0.5',
        mode='t1_zw',
        split='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/t1_zw_fse_val.txt',
        pipeline=test_pipeline,
        ),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
                    'hie_niigz_mask_seg_0.5x0.5x0.5',
        ann_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
                    'hie_niigz_mask_seg_0.5x0.5x0.5',
        mode='t1_zw',
        split='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/t1_zw_fse_val.txt',
        pipeline=test_pipeline,))
evaluation = dict(interval=2, metric=['accuracy', 'precision', 'recall', 'f1_score', 'support', 'auc'])


norm_cfg = dict(type='BN3d', requires_grad=True)
conv_cfg = dict(type='Conv3d')
num_classes = 2
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetV1d',
        depth=18,
        in_channels=1,
        in_dims=3,
        num_stages=4,
        out_indices=(3,),
        style='pytorch',
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        init_cfg=[
             dict(type='Kaiming', layer=['Conv3d']),
             dict(
                 type='Constant',
                 val=1,
                 layer=['_BatchNorm', 'GroupNorm', 'BN3d'])
         ],
        train_cfg=dict(augments=[
                dict(type='BatchMixup', alpha=0.8, num_classes=num_classes, prob=0.5),
                dict(type='BatchCutMix', alpha=1.0, num_classes=num_classes, prob=0.5)
            ])
    ),
    neck=dict(type='GlobalAveragePooling', dim=3),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1,),
    ),
)

optimizer = dict(
    type='AdamW',
    lr=5e-4 * 128 * 8 / 512,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),)
optimizer_config = dict(grad_clip=dict(max_norm=5.0))
# learning policy
# lr_config = dict(policy='step', step=[40, 80, 120])
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20 * 1252,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=500)

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
load_from = '/opt/data/private/project/charelchen.cj/workDir/project/mmsegmentation-3D/' \
            'work_dirs/resnetV1d18_basecfg_epoch_runner_4Channel_dice_bce/iter_20000.pth'
resume_from = None
workflow = [('train', 1)]
