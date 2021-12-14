# [Learning to Detect Every Thing in an Open World (LDET)](https://arxiv.org/pdf/2112.01698.pdf)

<center>
<img src="docs/images/ldet_teaser.gif" width="70%"> </a></center>

**[[Paper]](https://arxiv.org/pdf/2112.01698.pdf) | [[Project Page]](https://ksaito-ut.github.io/openworld_ldet/) | [[Demo Video]](https://cs-people.bu.edu/keisaito/videos/video_let/video2_concat.mp4)**

If you use this code for your research, please cite:

Learning to Detect Every Thing in an Open World.
Kuniaki Saito, Ping Hu, Trevor Darrell, Kate Saenko. In Arxiv 2021. [[Bibtex]](docs/bib.txt)

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
* Install PyTorch. You can choose the PyTorch and CUDA version according to your machine. Just make sure your PyTorch version matches the prebuilt Detectron2 version (next step). Example for PyTorch v1.6.0:
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
We provide evaluation on COCO in this repository:
- [COCO](http://cocodataset.org/)

Note that we will release dataset description for UVO, Cityscape, and Mapillary upon acceptance.
But, we provide the configuration files for reference.
Please follow the standard data preparation for detectron2.


## Getting Started

### Training & Evaluation in Command Line

To train a model, run
```angular2html
sh run_train.sh configs/VOC-COCO/voc_coco_mask_rcnn_R_50_FPN.yaml save_dir
```

To evaluate the trained models, run
```angular2html
sh run_test.sh configs/VOC-COCO/voc_coco_mask_rcnn_R_50_FPN.yaml weight_to_eval
```
