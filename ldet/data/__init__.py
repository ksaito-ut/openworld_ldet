from .builtin import register_all_coco, register_all_lvis, register_all_cityscapes
from . import builtin as _builtin  # ensure the builtin datasets are registered
from .copy_paste_mapper import CopyPasteMapper
from .build import (
    build_batch_data_loader,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    load_proposals_into_dataset,
    print_instances_class_histogram,
    filter_images_with_only_crowd_annotations,
)

__all__ = [k for k in globals().keys() if not k.startswith("_")]
