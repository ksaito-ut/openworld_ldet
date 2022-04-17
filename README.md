## [Learning to Detect Every Thing in an Open World](https://arxiv.org/pdf/2112.01698.pdf)

<p align="center">
<img src="docs/images/ldet_teaser.gif" width="70%">
</p>

**[[Paper]](https://arxiv.org/pdf/2112.01698.pdf) | [[Project Page]](https://ksaito-ut.github.io/openworld_ldet/) | [[Demo Video]](https://cs-people.bu.edu/keisaito/videos/video_let/video2_concat.mp4)**

If you use this code for your research, please cite:

<em>Learning to Detect Every Thing in an Open World.

Kuniaki Saito, Ping Hu, Trevor Darrell, Kate Saenko. In Arxiv 2021. </em> [[Bibtex]](https://github.com/ksaito-ut/openworld_ldet/docs/bib.txt)

## News
[2022/02/20] We find a bug in evaluation code and are rewriting the paper. Please contact the authors if you plan to cite our papers. 

[2022/03/11] We fixed the bug in ldet/evaluation/coco_evaluation.py. The pascal class id was not indexed correctly. We will further update the repo following our updated paper.

[2022/04/11] Updated the evaluation code following new [paper](docs/paper.pdf).

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
The json file split used for evaluation is available in [Dropbox Link](https://drive.google.com/file/d/1bn4oIdV53xVTPfp9BG9dplCcpZ6Yz3hR/view?usp=sharing)
Update the line in [builtin.py](https://github.com/ksaito-ut/openworld_ldet/blob/884cbd1eec347f7c1d0bd36bba0c2b1cc5c2cdc4/ldet/data/builtin.py#L34).


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
- [Cityscapes](https://www.cityscapes-dataset.com/login/). Follow [detectron2's instruction](https://github.com/facebookresearch/detectron2/tree/main/datasets). Update the line in [builtin.py](https://github.com/ksaito-ut/openworld_ldet/blob/884cbd1eec347f7c1d0bd36bba0c2b1cc5c2cdc4/ldet/data/builtin.py#L103).

- [Mapillary](https://www.mapillary.com/dataset/vistas).
Update the line in [builtin.py](https://github.com/ksaito-ut/openworld_ldet/blob/884cbd1eec347f7c1d0bd36bba0c2b1cc5c2cdc4/ldet/data/builtin.py#L37).

- [Obj365](https://open.baai.ac.cn/data-set-detail/MTI2NDc=/MTA=/true).
We used [validation set](https://open.baai.ac.cn/data-set-detail/MTI2NDc=/MTA=/true#:~:text=zhiyuan_objv2_val.-,json,-%E8%AE%AD%E7%BB%83%E9%9B%86).
We pick 5000 images for evaluation. The json is available [here](https://drive.google.com/file/d/1WAeVviaNZTMUNnOfdhOUQWoRajT3Zc6V/view?usp=sharing).
Please rewrite the path of the lists following your environment.


## Trained models
The trained weights are available from link attached with <a>model</a>.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Method</th>
<th valign="center">Training Dataset</th>
<th valign="center">Evaluation Dataset</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">box<br/>AR</th>
<th valign="bottom">seg<br/>AP</th>
<th valign="bottom">seg<br/>AR</th>
<th valign="center">Link</th>
<!-- TABLE BODY -->
<!-- ROW: Plain -->
 <tr><td align="center"><a href="tools/trainer_plain.py">Mask RCNN</a></td>
<td align="center">VOC-COCO</td>
<td align="center">Non-VOC</td>
<td align="center">1.5</td>
<td align="center">10.9</td>
<td align="center">0.7</td>
<td align="center">9.1</td>
<td align="center"><a href="https://drive.google.com/file/d/1-38ADOW0T4dfd7BhhCm_Fro9_WoB_nH-/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="configs/VOC-COCO/voc_coco_mask_rcnn_R_50_FPN.yaml">config</a></td>
<!-- ROW: Sampling -->
 <tr><td align="center"><a href="tools/trainer_plain.py">Mask RCNN<sup>S</sup></a></td>
<td align="center">VOC-COCO</td>
<td align="center">Non-VOC</td>
<td align="center">3.4</td>
<td align="center">18.0</td>
<td align="center">2.2</td>
<td align="center">15.8</td>
<td align="center"><a href="https://drive.google.com/file/d/1ooo0vLFC_LSZA_eDgMU1GNzsiN0FcqbI/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="configs/VOC-COCO/voc_coco_mask_rcnn_R_50_FPN.yaml">config</a></td>
<!-- ROW: LDET -->
<tr><td align="center"><a href="tools/trainer_ldet.py">LDET</a></td>
<td align="center">VOC-COCO</td>
<td align="center">Non-VOC</td>
<td align="center">5.0</td>
<td align="center">30.8</td>
<td align="center">4.7</td>
<td align="center">27.4</td>
<td align="center"><a href="https://drive.google.com/file/d/1TkoQw6ULKgpvqHAuvcCCPGa2Km6KEOj8/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="configs/VOC-COCO/voc_coco_mask_rcnn_R_50_FPN.yaml">config</a>
</td>
</tr>

<tr><td align="center"><a href="tools/trainer_plain.py">Mask RCNN<sup>S</sup></a></td>
<td align="center">COCO</td>
<td align="center">UVO</td>
<td align="center">25.3</td>
<td align="center">42.3</td>
<td align="center">20.6</td>
<td align="center">35.9</td>
<td align="center"><a href="https://drive.google.com/file/d/1LEgTJwBJw_tlqLqLwRix1kp-rgJb34Zy/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="configs/COCO/mask_rcnn_R_50_FPN.yaml">config</a>
</td>
</tr>


<tr><td align="center"><a href="tools/trainer_ldet.py">LDET</a></td>
<td align="center">COCO</td>
<td align="center">UVO</td>
<td align="center">25.8</td>
<td align="center">47.5</td>
<td align="center">21.9</td>
<td align="center">40.7</td>
<td align="center"><a href="https://drive.google.com/file/d/1A3CXQig95PR5sMA5IjhiqqxCT4lSQRi1/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="configs/COCO/mask_rcnn_R_50_FPN.yaml">config</a>
</td>
</tr>
</table>


## Training & Evaluation

### Training
To train a model, run
```angular2html
## Training on VOC-COCO
sh tools/run_train.sh configs/VOC-COCO/voc_coco_mask_rcnn_R_50_FPN.yaml save_dir
## Training on COCO
sh tools/run_train.sh configs/COCO/mask_rcnn_R_50_FPN.yaml save_dir
## Training on Cityscapes
sh tools/run_train.sh configs/Cityscapes/mask_rcnn_R_50_FPN.yaml save_dir

```
Note that the training will produce two directories, i.e., one for normal models and the other for exponential moving averaged models. We used the latter for evaluation.

### Evaluation
To evaluate the trained models, run
```angular2html
## Test on Non-VOC-COCO
sh tools/run_test.sh configs/VOC-COCO/voc_coco_mask_rcnn_R_50_FPN.yaml weight_to_eval
## Test on UVO, Obj365
sh tools/run_test.sh configs/COCO/mask_rcnn_R_50_FPN.yaml weight_to_eval
## Test on Mapillary
sh tools/run_test.sh configs/Cityscapes/mask_rcnn_R_50_FPN.yaml weight_to_eval

```

The above script will show two results: agnostic mode and classwise mode.
The agnostic mode regards all instances as a single class while classwise mode makes distinction on different classes.
To consider class imbalance, we report AR in classwise mode in our paper while reporting AP in agnostic mode.
Note that the above script computes performance on novel classes. To get performance on all classes, please disable the flag of exclude_known.