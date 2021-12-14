## [Learning to Detect Every Thing in an Open World](https://arxiv.org/pdf/2112.01698.pdf)

<p align="center">
<img src="docs/images/ldet_teaser.gif" width="70%">
</p>

**[[Paper]](https://arxiv.org/pdf/2112.01698.pdf) | [[Project Page]](https://ksaito-ut.github.io/openworld_ldet/) | [[Demo Video]](https://cs-people.bu.edu/keisaito/videos/video_let/video2_concat.mp4)**

If you use this code for your research, please cite:

<em>Learning to Detect Every Thing in an Open World.

Kuniaki Saito, Ping Hu, Trevor Darrell, Kate Saenko. In Arxiv 2021. </em> [[Bibtex]](https://github.com/ksaito-ut/openworld_ldet/docs/bib.txt)

## Installation

**Requirements**
* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/)
* [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
* CUDA

**Build LDET**
* Create a virtual environment.
We used `conda` to create a new environment.
```angular2html
conda create --name ldet
conda activate ldet
```
* Install PyTorch. You can choose the PyTorch and CUDA version according to your machine. Just make sure your PyTorch version matches the prebuilt Detectron2 version (next step). Example for PyTorch v1.10.0:
```angular2html
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
Currently, the codebase is compatible with [Detectron2 v0.6](https://github.com/facebookresearch/detectron2/releases/tag/v0.6). Example for PyTorch v1.10.0 and CUDA v11.3:
* Install Detectron2 v0.6
```angular2html
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```
* Install other requirements.
```angular2html
python3 -m pip install -r requirements.txt
```

## Code Structure
- **configs**: Configuration files
- **ldet**
  - **data**: Code related to dataset configuration.
  - **data/copy_paste_mapper.py**: Code for our data augmentation.
  - **engine**: Contains config file for training.
  - **evaluation**: Code used for evaluation.
  - **modeling**: Code for models, including backbones, prediction heads.
- **tools**
  - **trainer_copypaste.py**: Training and testing script.
  - **run_test.sh**: Evaluation script
  - **run_train.sh**: training script

## Data Preparation
We provide evaluation on COCO, UVO, and Mapillary (v2.0) in this repository:
- [COCO](http://cocodataset.org/).
Trained on train split, evaluated on validation split. Download the COCO dataset following the instruction of detectron2.

- [UVO](https://sites.google.com/view/unidentified-video-object/dataset?authuser=0).
We downloaded [uvo_videos_sparse.zip](https://drive.google.com/drive/folders/1fOhEdHqrp_6D_tBsrR9hazDLYV2Sw1XC) and evaluated on the videos. Follow their instructions to split videos into frames.
The split used for evaluation is available in [Dropbox Link](https://drive.google.com/file/d/1bn4oIdV53xVTPfp9BG9dplCcpZ6Yz3hR/view?usp=sharing)

E.g., the data structure of UVO dataset is as follows:
```angular2html
uvo_frames_sparse/video1/0.png
uvo_frames_sparse/video1/1.png
.
.
.
uvo_frames_sparse/video2/0.png
.
```
- [Cityscapes](https://www.cityscapes-dataset.com/login/) Follow [detectron2's instruction](https://github.com/facebookresearch/detectron2/tree/main/datasets).

- [Mapillary](https://www.mapillary.com/dataset/vistas)


## Getting Started

### Training & Evaluation in Command Line

To train a model, run
```angular2html
sh tools/run_train.sh configs/VOC-COCO/voc_coco_mask_rcnn_R_50_FPN.yaml save_dir
```

To evaluate the trained models, run
```angular2html
sh tools/run_test.sh configs/VOC-COCO/voc_coco_mask_rcnn_R_50_FPN.yaml weight_to_eval
```
