import numpy as np
import os, json, cv2, random
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

# setup logger
setup_logger()

# get config
cfg = get_cfg()

# add config
cfg_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # set threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)

# prepare predictor
predictor = DefaultPredictor(cfg)

# custom visualizer
class MyVisualizer(Visualizer):
    def _jitter(self, color):
        return color # never use jitter

# run Mask-RCNN
def run(image):
    # predict outputs
    outputs = predictor(image)

    # visualize results
    vis = MyVisualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.SEGMENTATION)
    out = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]
