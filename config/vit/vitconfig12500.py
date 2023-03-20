#VIT BASE https://huggingface.co/docs/transformers/model_doc/vit
custom_imports = dict(imports=['mmseg.datasets.mydata','mmseg.datasets.pipelines.mytransforms'], allow_failed_imports=False)
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VisionTransformer',
        img_size=(288,288),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(2, 5, 8, 11),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=False,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        interpolate_mode='bilinear'),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

dataset_type = 'FiberDataset'
data_root = ''
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomRotate', prob=0.5, degree=0.1),
    dict(type='RandomShift', max_shift_px=14),
    dict(type='Resize', img_scale=(288, 288), ratio_range=[0.95, 1.05]),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(288, 288),
        transforms=[
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='FiberDataset',
        data_root='/global/cfs/projectdirs/m636/avaylon/Fiber/Full/full/train',
        img_dir=
        '/global/cfs/projectdirs/m636/avaylon/Fiber/Full/full/train/img',
        ann_dir=
        '/global/cfs/projectdirs/m636/avaylon/Fiber/Full/full/train/ann',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='RandomRotate', prob=0.5, degree=0.1),
            dict(type='RandomShift', max_shift_px=14),
            dict(
                type='Resize', img_scale=(288, 288), ratio_range=[0.95, 1.05]),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        split='/global/homes/a/aj2000/vtfs/splits/train_exp1_12500.txt'),
    val=dict(
        type='FiberDataset',
        data_root='/global/cfs/projectdirs/m636/avaylon/Fiber/Full/full/val',
        img_dir='/global/cfs/projectdirs/m636/avaylon/Fiber/Full/full/val/img',
        ann_dir='/global/cfs/projectdirs/m636/avaylon/Fiber/Full/full/val/ann',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(288, 288),
                transforms=[
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='/global/homes/a/aj2000/vtfs/splits/val_exp1_2500.txt'),
    test=dict(
        type='FiberDataset',
        data_root='global/homes/a/aj2000/vtfs',
        img_dir='/global/homes/a/aj2000/vtfs/imgs',
        ann_dir='/global/homes/a/aj2000/vtfs/annot',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(288, 288),
                transforms=[
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='/global/u1/a/aj2000/vtfs/testsetsmall.txt'))


log_config = dict(
    interval=10, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

optimizer = dict(
    type='AdamW', lr=0.00001, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = dict()

lr_config = dict(
    policy='poly',
    power=1.0,
    min_lr=0.000001,
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=15625)
checkpoint_config = dict(
    by_epoch=False, interval=15625, meta=dict(CLASSES=2, PALETTE=None))

evaluation = dict(interval=100, metric=['mIoU', 'mDice'], pre_eval=True)
work_dir = '/global/cfs/projectdirs/m636/AJ/p1_models/vit/vit_12500_bsize2'
seed = 0
gpu_ids = range(0, 4)

