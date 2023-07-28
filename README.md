# InternImage for Semantic Segmentation

This folder contains the implementation of the InternImage for semantic segmentation. 

Our segmentation code is developed on top of [MMSegmentation v0.27.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.27.0).

## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/PerfectDreamComeTrue/Building-Segmentation-on-Satellite-Images-Internimage.git
cd InternImage
```

- Create a conda virtual environment and activate it:

```bash
conda create -n internimage python=3.7 -y
conda activate internimage
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install torch==1.11 with CUDA==11.3:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

- Install `timm==0.6.11` and `mmcv-full==1.6.0`:

```bash
pip install -U openmim
!pip install mmcv-full==1.6.0
!pip install timm==0.6.11 mmdet==2.28.1
!pip install mmengine
```

- Install `mmsegmentation==0.27.0`:
```bash
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v0.27.0
pip install -v -e .
pip install mmsegmentation
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
```

- Compile CUDA operators
```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
- You can also install the operator using .whl files
[DCNv3-1.0-whl](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files)

### Data Preparation

I use Cityscapes format
```bash
Cityscapes
  └───leftImg8bit
  │   └───train
  │   │   └───{image_id}_leftImg8bit.png   
  │   └───val
  │   │   └───{image_id}_leftImg8bit.png
  │   └───test
  │       └───{image_id}_leftImg8bit.png
  └───gtFine
      └───train
      │   └───{image_id}_gtFine_TrainlabelIds.png
      └───val
          └───{image_id}_gtFine_TrainlabelIds.png
```

### Evaluation

To evaluate `InternImage` on `Cicyscapes_custum` val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval mIoU
```

For example, to evaluate the `InternImage-T` with a `single GPU`:

```bash
python test.py configs/cityscapes/upernet_internimage_t_512x1024_40k_cityscapes_custom.py checkpoint_dir/seg/upernet_internimage_t_512x1024_40k_cityscapes_custom.pth --eval mIoU
```

For example, to evaluate the `InternImage-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/cityscapes/upernet_internimage_t_512x1024_40k_cityscapes_custom.py checkpoint_dir/seg/upernet_internimage_t_512x1024_40k_cityscapes_custom.pth 8 --eval mIoU
```

### Training

To train an `InternImage` on `Cicyscapes_custum`, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `InternImage-T` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/cityscapes/upernet_internimage_t_512x1024_40k_cityscapes_custom.py 8
```


### Image Demo
To inference a single image like this:
```
CUDA_VISIBLE_DEVICES=0 python image_demo.py \
  data/ade/ADEChallengeData2016/images/validation/ADE_val_00000591.jpg \
  configs/ade20k/upernet_internimage_t_512_160k_ade20k.py  \
  checkpoint_dir/seg/upernet_internimage_t_512_160k_ade20k.pth  \
  --palette ade20k 
```

### Export

To export a segmentation model from PyTorch to TensorRT, run:
```shell
MODEL="model_name"
CKPT_PATH="/path/to/model/ckpt.pth"

python deploy.py \
    "./deploy/configs/mmseg/segmentation_tensorrt_static-512x512.py" \
    "./configs/ade20k/${MODEL}.py" \
    "${CKPT_PATH}" \
    "./deploy/demo.png" \
    --work-dir "./work_dirs/mmseg/${MODEL}" \
    --device cuda \
    --dump-info
```

For example, to export `upernet_internimage_t_512_160k_ade20k` from PyTorch to TensorRT, run:
```shell
MODEL="upernet_internimage_t_512_160k_ade20k"
CKPT_PATH="/path/to/model/ckpt/upernet_internimage_t_512_160k_ade20k.pth"

python deploy.py \
    "./deploy/configs/mmseg/segmentation_tensorrt_static-512x512.py" \
    "./configs/ade20k/${MODEL}.py" \
    "${CKPT_PATH}" \
    "./deploy/demo.png" \
    --work-dir "./work_dirs/mmseg/${MODEL}" \
    --device cuda \
    --dump-info
```
