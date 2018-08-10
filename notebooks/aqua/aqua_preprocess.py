import random
import os
import shutil
import copy
from skimage import measure
from skimage import morphology as skim
import skimage.io as skio
import warnings
import pandas as pd
import numpy as np
import pathlib
import yaml
import geopandas as gpd
from rasterio import features, coords
import rasterio
from shapely.geometry import shape
import gdal

random.seed(1)

# Directory setup
ROOT = "/home/tclavelle/tana-crunch/CropMask_RCNN"
DATASET = os.path.join(ROOT, 'data/aqua')
PREPPED = os.path.join(DATASET, 'prepped_planet')
TRAIN = os.path.join(DATASET, 'train')
TEST = os.path.join(DATASET, 'test')
RESULTS = os.path.join(DATASET, 'results')

def make_dirs():
    
    dirs = [DATASET, PREPPED, TRAIN, TEST, RESULTS]

    # Make directory and subdirectories
    for d in dirs:
        pathlib.Path(d).mkdir(parents=False, exist_ok=False)

# Class dictionary
AQUA_CLASS = {"line": 1, "raft": 2, "pond": 3, "cage": 4}
        
    
if __name__ == '__main__':
    print('sourced preprocess module')