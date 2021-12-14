# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN


def add_copypaste_config(cfg):
    """
    Add config for CopyPaste.
    """
    cfg.DATASETS.BACK_RATIO = 8
