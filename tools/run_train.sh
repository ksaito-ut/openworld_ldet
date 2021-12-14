str_ema="${1}_ema"
mkdir $str_ema
python trainer_ldet.py --config ../configs/VOC-COCO/voc_coco_mask_rcnn_R_50_FPN.yaml --num-gpus 2 OUTPUT_DIR $1

