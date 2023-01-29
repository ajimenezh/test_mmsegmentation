import numpy as np

# Let's take a look at the dataset
#import mmcv
import matplotlib.pyplot as plt
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import copy
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv import Config
from mmseg.apis import set_random_seed
from mmseg.utils import get_device
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor

from PIL import Image

#img_dir = r"inputs"
#data_root =  r"C:\\Users\\Alex\\Downloads\\suadd_23-v0.1\\"
#ann_dir = r'semantic_annotations'

palette = ([ 148, 218, 255 ],  # light blue
        [  85,  85,  85 ],  # almost black
        [ 200, 219, 190 ],  # light green
        [ 166, 133, 226 ],  # purple    
        [ 255, 171, 225 ],  # pink
        [  40, 150, 114 ],  # green
        [ 234, 144, 133 ],  # orange
        [  89,  82,  96 ],  # dark gray
        [ 255, 255,   0 ],  # yellow
        [ 110,  87, 121 ],  # dark purple
        [ 205, 201, 195 ],  # light gray
        [ 212,  80, 121 ],  # medium red
        [ 159, 135, 114 ],  # light brown
        [ 102,  90,  72 ],  # dark brown
        [ 255, 255, 102 ],  # bright yellow
        [ 251, 247, 240 ])  # almost white

class TestSegmentationModel:
    def __init__(self):
        """
        Initialize your model here
        """
        self.class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255]

        #self.cfg = Config.fromfile('configs/convnext/upernet_convnext_tiny_fp16_512x512_160k_ade20k.py')
        self.cfg = Config.fromfile('configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py')

        # Since we use only one GPU, BN is used instead of SyncBN
        self.cfg.norm_cfg = dict(type='BN', requires_grad=True)

        # Set up working dir to save files and logs.
        self.cfg.work_dir = './work_dirs'
        #self.cfg.img_dir = img_dir
        #self.cfg.data_root =  data_root
        #self.cfg.ann_dir = ann_dir

        # Set seed to facitate reproducing the result
        self.cfg.seed = 0
        set_random_seed(0, deterministic=False)
        self.cfg.gpu_ids = range(1)
        self.cfg.device = get_device()

        # Build the detector
        self.model = build_segmentor(self.cfg.model)
        # Add an attribute for visualization convenience
        self.model.CLASSES = ( "WATER",
                "ASPHALT",
                "GRASS",
                "HUMAN",
                "ANIMAL",
                "HIGH_VEGETATION",
                "GROUND_VEHICLE",
                "FACADE",
                "WIRE",
                "GARDEN_FURNITURE",
                "CONCRETE",
                "ROOF",
                "GRAVEL",
                "SOIL",
                "PRIMEAIR_PATTERN",
                "SNOW")

        self.cfg.model.pretrained = None
        # Enable test time augmentation
        self.cfg.data.test.pipeline[1].flip = True

        self.checkpoint_file = self.cfg.work_dir + "/latest.pth"

        self.model.cfg = self.cfg
        self.model = init_segmentor(self.cfg, self.checkpoint_file, device='cuda')
    
    def segment_single_image(self, image_to_segment):
        """
        Implements a function to segment a single image
        Inputs:
            image_to_segment - Single frame from onboard the flight

        Outputs:
            An 2D image with the pixels values corresponding to the label number
        """
        img = np.array(Image.fromarray(image_to_segment).convert('RGB'))
        #img = mmcv.imread(img)
        result = inference_segmentor(self.model, img)
        return result[0]
