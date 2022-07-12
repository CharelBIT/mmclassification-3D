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
        # img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
        #             'hie_niigz_mask_seg_0.5x0.5x0.5',
        # ann_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
        #             'hie_niigz_mask_seg_0.5x0.5x0.5',
        img_dir='/gruntdata/workDir/dataset/hie/'
                  'ertonghos_nifty_data_0.5x0.5x0.5',
        ann_dir='/gruntdata/workDir/dataset/hie/'
                  'ertonghos_nifty_data_0.5x0.5x0.5',
        mode='t2',
        # split='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/t1_zw_fse_train.txt',
        split='/gruntdata/workDir/dataset/hie/ertong_list.txt',
        pipeline=train_pipeline,
        ),
    val=dict(
        type=dataset_type,
        # img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
        #             'hie_niigz_mask_seg_0.5x0.5x0.5',
        # ann_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
        #             'hie_niigz_mask_seg_0.5x0.5x0.5',
        img_dir='/gruntdata/workDir/dataset/hie/'
                  'ertonghos_nifty_data_0.5x0.5x0.5',
        ann_dir='/gruntdata/workDir/dataset/hie/'
                  'ertonghos_nifty_data_0.5x0.5x0.5',
        mode='t2',
        split='/gruntdata/workDir/dataset/hie/ertong_list.txt',
        pipeline=test_pipeline,
        ),
    # test=dict(
    #     # replace `data/val` with `data/test` for standard test
    #     type=dataset_type,
    #     img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
    #                 'extend_validation_nifty_0.5x0.5x0.5',
    #     ann_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/'
    #                 'extend_validation_nifty_0.5x0.5x0.5',
    #     mode='t1_zw',
    #     split='/opt/data/private/project/charelchen.cj/workDir/dataset/hie/extend_datalist.txt',
    #     pipeline=test_pipeline,))
    test=[])
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
         ]
    ),
    neck=dict(type='GlobalAveragePooling', dim=3),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
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
load_from = '/opt/data/private/project/charelchen.cj/workDir/project/mmsegmentation-3D/' \
            'work_dirs/resnetV1d18_basecfg_epoch_runner_4Channel_dice_bce_t2/iter_20000.pth'
resume_from = None
workflow = [('train', 1)]
