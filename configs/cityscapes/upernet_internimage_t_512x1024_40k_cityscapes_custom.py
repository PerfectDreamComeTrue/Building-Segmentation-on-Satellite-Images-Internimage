# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/upernet_r50.py', #custom
    '../_base_/datasets/cityscapes_custom.py', #custom
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_40k.py'   #custom: 160k -> 40k
]
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/upernet_internimage_t_512x1024_160k_cityscapes.pth' #custom: cityscapes pretrian 160k
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=64,
        depths=[4, 4, 18, 4],
        groups=[4, 8, 16, 32],
        mlp_ratio=4.,
        drop_path_rate=0.2,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=1.0,
        post_norm=False,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(num_classes=1, in_channels=[64, 128, 256, 512]), # custom: num_classes
    auxiliary_head=dict(num_classes=1, in_channels=256), # custom num_classes
    test_cfg=dict(mode='whole')
)
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=30, layer_decay_rate=1.0,
                       depths=[4, 4, 18, 4]))
# learning policy
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
runner = dict(type='IterBasedRunner', max_iters=35700) #7140 총데이터 / sampler_per_gpu =3570
checkpoint_config = dict(by_epoch=False, interval=3570, max_keep_ckpts=1)
evaluation = dict(interval=3570, metric='mIoU', save_best='mIoU') # custom: interval 16000>2000
# fp16 = dict(loss_scale=dict(init_scale=512))
