"""
Detectron2 training script with a copy-paste augmentation training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in LDET.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use LDET as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import copy
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
)
from detectron2.evaluation import DatasetEvaluators, print_csv_format
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
import sys
sys.path.append(os.getcwd().replace("tools", ""))
from ldet.engine import add_copypaste_config
from ldet.data import CopyPasteMapper, build_detection_test_loader, build_detection_train_loader
from ldet.evaluation import COCOEvaluator,  inference_on_dataset
from ldet.modeling import build_model
logger = logging.getLogger("openworld")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, tasks=("bbox", "segm") if 'obj365' not in dataset_name else ("bbox", )))
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, args):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        logger.info("Running inference with {}".format(dataset_name))
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        eval_obj365 = True if "obj365" in dataset_name else False
        results_i = inference_on_dataset(model,
                                         data_loader,
                                         evaluator,
                                         agnostic=True,
                                         exclude_known=args.exclude_known if 'uvo' not in dataset_name else False,
                                         classwise_mode=args.classwise_mode if 'uvo' not in dataset_name else False,
                                         eval_obj365=eval_obj365)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def update_average(model_tgt, model_src, beta=0.999):
    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())

        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


def do_train(cfg, model, resume=False):

    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    ## We save moving average models and test on them.
    model_ema = copy.deepcopy(model)
    os.makedirs(cfg.OUTPUT_DIR+"_ema", exist_ok=True)
    checkpointer_ema = DetectionCheckpointer(
        model_ema, cfg.OUTPUT_DIR+"_ema", optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    _ = (
        checkpointer_ema.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    periodic_checkpointer_ema = PeriodicCheckpointer(
        checkpointer_ema, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    data_loader = build_detection_train_loader(cfg, mapper=CopyPasteMapper(cfg, is_train=True))
    logger.info("Starting training from iteration {}".format(start_iter))
    weight_dict = {"loss_rpn_cls": 0.0,
                   "loss_rpn_loc": 0.0,
                   "loss_cls": 0.0,
                   "loss_box_reg": 0.0,
                   "loss_mask": 1.0,
                   }
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            loss_dict_normal = model(data)
            loss_dict_normal = {k: v * weight_dict[k] for k, v in loss_dict_normal.items()}
            for d in data:
                d['image'] = d['copy_image']
            loss_dict_paste = model(data)
            loss_dict_paste = {k: v * (1 - weight_dict[k]) for k, v in loss_dict_paste.items()}
            losses = sum(loss_dict_normal.values())
            losses += sum(loss_dict_paste.values())
            assert torch.isfinite(losses).all(), loss_dict_paste
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict_normal).items()}
            for k, v in comm.reduce_dict(loss_dict_paste).items():
                loss_dict_reduced[k + "_paste"] = v.item()
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            update_average(model_ema, model, beta=1-optimizer.param_groups[0]["lr"])
            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
            periodic_checkpointer_ema.step(iteration)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_copypaste_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model, args)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--classwise-mode", default=False, action='store_true', help="class wise output")
    parser.add_argument("--exclude-known", default=False, action='store_true', help="novel class evaluation")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
