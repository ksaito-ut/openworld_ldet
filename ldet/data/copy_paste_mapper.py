import copy
import logging
import numpy as np
import cv2
import random
from typing import List, Optional, Union
import torch
from detectron2.config import configurable
import os
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import pycocotools.mask as mask_util
import torch.nn.functional as nnf
from skimage.filters import gaussian


"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["CopyPasteMapper"]


class CopyPasteMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        back_ratio: int = 8,
        sample_texture: bool=False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
            back_ratio: to define the size background region to crop from image.
            (width / back_ratio, height/back_ratio) is randomly cropped.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.back_ratio = back_ratio
        self.sample_texture = sample_texture
        ## path to texture images if available.
        if sample_texture:
            self.path_to_back = "path_to_DTD"
            self.list_backs = make_dataset(self.path_to_back)

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False
        ## back_ratio defines the size of background region.
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "back_ratio": int(cfg.DATASETS.BACK_RATIO)
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret


    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)


    def random_texture(self, image, width=256, height=256):

        sample_back = random.randint(0, len(self.list_backs)-1)
        back_image = utils.read_image(self.list_backs[sample_back], format=self.image_format)
        width = min(back_image.shape[0], width)
        height = min(back_image.shape[1], height)
        width_start = random.randint(0, max(back_image.shape[0] - width, 1))
        height_start = random.randint(0, max(back_image.shape[1] - height, 1))
        back_image = back_image.transpose((2, 0, 1))
        background = back_image[:, width_start:width_start + width,
                     height_start:height_start + height]
        background = torch.from_numpy(background.copy()).float()
        return nnf.interpolate(background.view(1, 3, background.shape[1], background.shape[2]),
                               size=(image.size(1), image.size(2)), mode='bilinear')


    def copypaste(self, dataset_dict):
        # function to apply copy-paste augmentation.
        d = dataset_dict
        masks = d["instances"].get_fields()['gt_masks']
        image = d['image']
        height, width = d["instances"]._image_size
        # get polygon mask
        polygons = masks.polygons
        array_polygon = [np.asarray(y).reshape(-1) for x in polygons for y in x]
        tmp = mask_util.frPyObjects(array_polygon, height, width)
        tmp = mask_util.merge(tmp)
        mask_final = mask_util.decode(tmp)[:, :]
        # Apply blurring to mask as well as input image.
        mask_final = gaussian(mask_final, sigma=1.0, preserve_range=True)
        image = gaussian(image.numpy(), sigma=1.0, preserve_range=True)
        image = torch.from_numpy(image)
        # choose region to crop for background.
        if self.sample_texture:
            back_region = self.random_texture(image, width=256, height=256)
        else:
            back_region = random_region(image, width=int(width // self.back_ratio),
                                        height=int(height // self.back_ratio))
            # resize image to avoid foreground images dissimilar from background.
            image_small = nnf.interpolate(image.view(1, 3, image.size(1), image.size(2)),
                                          size=(image.size(1) // self.back_ratio,
                                                image.size(2) // self.back_ratio),
                                          mode='bilinear')
            image = nnf.interpolate(image_small,
                                    size=(image.size(1),
                                          image.size(2)),
                                    mode='bilinear')
        # get masked image
        img_final = back_region * (1 - mask_final) + image[0] * mask_final
        kernel = np.ones((5, 5), np.float32) / 25
        # apply smoothing
        dst = cv2.filter2D(img_final[0].numpy().transpose(1, 2, 0), -1, kernel)
        dst = cv2.filter2D(dst, -1, kernel)
        d['copy_image'] = torch.from_numpy(dst.transpose(2, 0, 1)).float()
        return d


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return self.copypaste(dataset_dict)

def random_region(image, width=32, height=32):
    ## randomly crop region with size (width, height), return after resizing.

    width_start = random.randint(0, max(image.shape[1]-width, 1))
    height_start = random.randint(0, max(image.shape[2]-height, 1))
    image_crop = image[:, width_start:width_start+width,
                 height_start:height_start+height]
    return nnf.interpolate(image_crop.view(1, 3, image_crop.shape[1], image_crop.shape[2]),
                           size=(image.size(1), image.size(2)), mode='bilinear')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = path
                    images.append(item)
    return images

