# Vision Transformers for Segmentation of Ceramic Matrix Fibers:
Repository for the developments related to Transformers for Science.

Based on [MMsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/63f83258469cbe324143af51da4397aa3223dfd0).
Work done by Ajinkya Chaudhari at LBNL.



## Environment Setup: 
*All experiments were done on NERSC Perlmutter

1. Create a conda environment 
- `conda create --name VTFS_env python=3.7`

2. Install dependencies
- `pip install -r requirements.txt`

3. Clone this repository:
- `git clone https://github.com/ajinkya-ch/AJmmsegmentation.git`



## Training:

Eg: Vision Transformer (ViT) on 12500 data using 4 GPUs: `sh ./src/mmsegmentation/tools/dist_train.sh ./config/vit/vitconfig12500.py 4 `

Refer to this [link](https://mmsegmentation.readthedocs.io/en/latest/train.html) for more information.

## Inference:

Eg: Using Vision Transformer (ViT) trained on 12500 data to infer on test data using 4 GPUs: `sh ./src/mmsegmentation/tools/dist_test.sh ./config/vit/vitconfig12500.py {path_to_pretrained_model} 4 --eval mDice`

Refer to this [link](https://mmsegmentation.readthedocs.io/en/latest/inference.html) for more information.


## Models:


| name  | config | log  | pretrained model  |
|---|---|---|---|
| vit-12500  | [cfg](https://github.com/LBNL-CRD-MLA/VTFS/blob/main/config/vit/vitconfig12500.py)  | [log](https://github.com/LBNL-CRD-MLA/VTFS/blob/main/models/logs/vit12500.json) | [model](https://drive.google.com/file/d/1qbgpBV_b0Y96-qybvE7dUaezVausJUmj/view?usp=share_link)  |
|  vit-25000 | [cfg](https://github.com/LBNL-CRD-MLA/VTFS/blob/main/config/vit/vitconfig25k.py)  | [log](https://github.com/LBNL-CRD-MLA/VTFS/blob/main/models/logs/vit25000.json)  | [model](https://drive.google.com/file/d/1YDMTDo3XRxa-5sghV9VohVhUZJrZinJ1/view?usp=share_link)  |
| vit-50000  | [cfg](https://github.com/LBNL-CRD-MLA/VTFS/blob/main/config/vit/vitconfig50k.py)  | [log](https://github.com/LBNL-CRD-MLA/VTFS/blob/main/models/logs/vit50000.json)  | [model](https://drive.google.com/file/d/18qjcdB7D2asG34e8MmNW4IQrEwSt_XTX/view?usp=share_link)  |
| swin-12500  | [cfg](https://github.com/LBNL-CRD-MLA/VTFS/blob/main/config/swin/swinconfig12500.py)  | [log](https://github.com/LBNL-CRD-MLA/VTFS/blob/main/models/logs/swin12500.json)  | [model](https://drive.google.com/file/d/1NxOZsCDK92okPZiaw9QUj6etfbjIklgA/view?usp=share_link)  |
| swin-25000  | [cfg](https://github.com/LBNL-CRD-MLA/VTFS/blob/main/config/swin/swinconfig25k.py)  | [log](https://github.com/LBNL-CRD-MLA/VTFS/blob/main/models/logs/swin25000.json)  | [model](https://drive.google.com/file/d/1ehZsHWcI5eyq9pr21XQf8Vk5x0Ew5E4Y/view?usp=share_link)  |
|  swin-50000 | [cfg](https://github.com/LBNL-CRD-MLA/VTFS/blob/main/config/swin/swinconfig50k.py)  | [log](https://github.com/LBNL-CRD-MLA/VTFS/blob/main/models/logs/swin50000.json)  | [model](https://drive.google.com/file/d/14fY3VA6_hk0ldzcs_DXeRsQ8ORehEs-l/view?usp=share_link)  |


## Results:


|   | vit-12500  | vit-25000  | vit-50000  | swin-12500   | swin-25000   | swin-50000   |
|---|---|---|---|---|---|---|
| testset1 (_6)| 83.15 |  83.59 | 84.08  |  83.78 | 84.0  | 83.6  |
| testset2 (_7x)|   |   |   |   |   |   |
| testset3 (_9) |   |   |   |   |   |   |
| testset4 (_10)|  83.55 | 83.09  |  82.93 |  82.18 |  82.31 | 80.47  |


## Known Issues:

1. KeyError: 'flip' : Solve by refering to this [solution](https://github.com/open-mmlab/mmsegmentation/issues/231)

2. No such file found ../Image.tiff : Solve by editing the suffix in ./src/mmseg/datasets/mydata.py

