# dataset settings
dataset_type = 'WUHANZL_ProstateDataset'
train_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type="LoadAnnotationsFromNIIFile"),
    dict(type='ExtractDataFromObj'),
    dict(type='NormalizeMedical', norm_type='full_volume_mean',
         instensity_min_val=0.5,
         instensity_max_val=99.5),
    dict(type='CropMedicalWithAnnotations', pad_mode='relative', relative_ratio=(0.05, 0.05, 0.05)),
    dict(type='ResizeMedical', size=(64, 64, 64)),
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
    dict(type='NormalizeMedical', norm_type='full_volume_mean',
         instensity_min_val=0.5,
         instensity_max_val=99.5),
    dict(type='CropMedicalWithAnnotations', pad_mode='relative', relative_ratio=(0.05, 0.05, 0.05)),
    dict(type='ResizeMedical', size=(64, 64, 64)),
    dict(type='ConcatImage'),
    dict(type='ToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/'
                'data_resample_100_1x1x1',
        img_suffixes=['image.nii.gz'],
        seg_map_suffix="mask.nii.gz",
        mode='T1WI',
        ann_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/'
                'data_resample_100_1x1x1',
        split='/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/'
              'trainlist_classifcation_100.txt',
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/'
                'data_resample_100_1x1x1',
        img_suffixes=['image.nii.gz'],
        seg_map_suffix="mask.nii.gz",
        mode='T1WI',
        ann_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/'
                'data_resample_100_1x1x1',
        split='/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/'
              'testlist_classifcation_100.txt',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/'
                'data_resample_100_1x1x1',
        img_suffixes=['image.nii.gz'],
        seg_map_suffix="mask.nii.gz",
        mode='T1WI',
        ann_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/'
                'data_resample_100_1x1x1',
        split='/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/'
              'testlist_classifcation_100.txt',
        pipeline=test_pipeline,
    )
)
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

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[40, 80, 120])
runner = dict(type='EpochBasedRunner', max_epochs=160)

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
load_from = None
resume_from = None
workflow = [('train', 1)]
