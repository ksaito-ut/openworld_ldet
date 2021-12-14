CUDA_VISIBLE_DEVICES=$2 python trainer_ldet.py  --config ../configs/VOC-COCO/voc_coco_mask_rcnn_R_50_FPN.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS $1

