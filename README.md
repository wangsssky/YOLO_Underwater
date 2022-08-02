# YOLO_Underwater
This is the official code for our paper "ULO: An Underwater Light-weight Object Detector for Edge Computing",
If you are interested in our work, please consider citing the following:
```
@Article{machines10080629,
AUTHOR = {Wang, Lin and Ye, Xiufen and Wang, Shunli and Li, Peng},
TITLE = {ULO: An Underwater Light-Weight Object Detector for Edge Computing},
JOURNAL = {Machines},
VOLUME = {10},
YEAR = {2022},
NUMBER = {8},
ARTICLE-NUMBER = {629},
URL = {https://www.mdpi.com/2075-1702/10/8/629},
ISSN = {2075-1702},
DOI = {10.3390/machines10080629}
}

```

## Introduction
This repo is based on
- [YOLO-v3](https://github.com/eriklindernoren/PyTorch-YOLOv3) 
- [pjreddie-darknet](https://github.com/pjreddie/darknet)  
- [YOLO-v5](https://github.com/ultralytics/yolov5)
- [YOLO Nano](https://github.com/wangsssky/YOLO-Nano)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [GhostNet](https://github.com/huawei-noah/Efficient-AI-Backbones#ghostnet-code)

## Project Structure
<pre>
│  main.py
│  README.md
│  test.py
│  train.py
│  val.py
│
├─data
|      image
|      box
│      test.txt
│      train.txt
│      urpc.names
│      val.txt
│
├─dataloader
│      data_split.py
│      URPCDataset.py
│
├─models
│  │  basic_layers.py
│  │  darknet_model.py
│  │  ghost_module.py
│  │  preprocessing_module.py
│  │  select_model.py
│  │  yolo_nano.py
│  │  yolo_nano_underwater.py
│  │  yolo_underwater.py
│  │  yolo_underwater_tiny.py
│  │
│  └─cfg
│          yolov3-tiny.cfg
│          yolov3.cfg
│          yolov4-tiny.cfg
│          yolov4.cfg
│
└─utils
        compute_anchor.py
        logger.py
        opts.py
        stats.py
        utils.py
</pre>

## Installation
```bash
git clone git@github.com:wangsssky/YOLO_Underwater.git
pip install -r requirements.txt
```

## Dataset
A optimized version of URPC2019 is used in the work, the updated annotations are available at https://github.com/wangsssky/Refined-training-set-of-URPC2019. 

## Train & Evaluate

train
```bash
python main.py --model YOLO-Underwater-Tiny --image_size 512  --num_epochs 300 
--batch_size 64 --lr 0.001 --num_threads 64 --gpu --weight_decay 5e-4 --preprocessing  
--checkpoint_path ./ckpt_YOLO_Underwater 

```
test
```bash
python main.py --model YOLO-Underwater --image_size 512  --batch_size 1 
--num_threads 4 --gpu  --test True --no_train --no_val --preprocessing
--resume_path ckpt_YOLO_Underwater/best.pth --conf_thresh 0.25 --nms_thresh 0.45
```


