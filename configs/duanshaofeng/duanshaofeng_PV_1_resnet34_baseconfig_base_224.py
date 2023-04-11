# dataset settings
dataset_type = 'ContinousSliceDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(
        type='Rotate',
        interpolation='bicubic',
        magnitude_key='angle',
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Brightness', magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Shear',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        direction='horizontal'),
    dict(
        type='Shear',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        direction='vertical'),
    dict(
        type='Translate',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        direction='horizontal'),
    dict(
        type='Translate',
        interpolation='bicubic',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        pad_val=tuple([round(x) for x in img_norm_cfg['mean'][::-1]]),
        direction='vertical')
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropWithAnnotation',
         expansion_type='statistic',
         expansion_kwargs={'expansion_val': 5, 'shift_aug': True}),
    dict(type='Resize', size=(224, 224)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies=policies,
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CropWithAnnotation',
         expansion_type='statistic',
         expansion_kwargs={'expansion_val': 5, 'shift_aug': False},
         # debug='/gruntdata1/charelchen.cj/workDir/dataset/whzn_lung/coco/CropWithAnnotation'
         ),
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
        img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/Deep-VMHCC/coco/PV/image',
        ann_file='/opt/data/private/project/charelchen.cj/workDir/dataset/Deep-VMHCC/coco/PV/annotation/instance_all.json',
        split='/opt/data/private/project/charelchen.cj/workDir/dataset/Deep-VMHCC/trainlist_2.txt',
        pipeline=train_pipeline,
        ),
    val=dict(
        type=dataset_type,
        img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/Deep-VMHCC/coco/PV/image',
        ann_file='/opt/data/private/project/charelchen.cj/workDir/dataset/Deep-VMHCC/coco/PV/annotation/instance_all.json',
        split='/opt/data/private/project/charelchen.cj/workDir/dataset/Deep-VMHCC/testlist_2.txt',
        pipeline=test_pipeline,
        ),
    test=[])
    # test=dict(
    #     # replace `data/val` with `data/test` for standard test
    #     type=dataset_type,
    #     img_dir='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/2dSlice_continous/image',
    #     ann_file='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/2dSlice_continous/annotation/instance_all.json',
    #     split='/opt/data/private/project/charelchen.cj/workDir/dataset/whtj_lung/test_split_130.txt',
    #     pipeline=test_pipeline,
    #     ))
evaluation = dict(interval=1, metric=['accuracy', 'precision', 'recall', 'f1_score', 'support', 'auc'])

# norm_cfg = dict(type='BN3d', requires_grad=True)
# conv_cfg = dict(type='Conv3d')
num_classes = 2
# model settings
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(
#         type='SwinTransformer', arch='small', img_size=224,
#         drop_path_rate=0.3),
#     neck=dict(type='GlobalAveragePooling', dim=1),
#     head=dict(
#         type='LinearClsHead',
#         num_classes=num_classes,
#         in_channels=768,
#         init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
#         loss=dict(
#             type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
#         cal_acc=False),
#     init_cfg=[
#         dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
#         dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
#     ],
#     train_cfg=dict(augments=[
#         dict(type='BatchMixup', alpha=0.8, num_classes=num_classes, prob=0.5),
#         dict(type='BatchCutMix', alpha=1.0, num_classes=num_classes, prob=0.5)
#     ]))

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        # frozen_stages=3,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1,),
    ),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=num_classes, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=num_classes, prob=0.5)
    ]))

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
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
    warmup_iters=940,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=160)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yPVf:enable
checkpoint_config = dict(by_epoch=True, interval=1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/opt/data/private/project/charelchen.cj/workDir/dataset/mmclassifcation_pretrained_model/' \
            'resnet34_batch256_imagenet_20200708-32ffb4f7.pth'
resume_from = None
workflow = [('train', 1)]
# work_dir = 'work_dirs/whzn_prostate_resnet18_baseconfig_base_224_continous_full_data_augment_T2'
