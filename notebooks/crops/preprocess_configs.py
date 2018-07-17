import numpy as np
import os

############################################################
#  Configurations
############################################################

class PPConfig():
    """Configuration for preprocessing imagery. 
    
    Stages include:
     - Create a working directory with subdirectories for intermediates and final gridded inputs
     - Select imagery channels, on season or off season or both, based on config selected, move
     - rasterize label vectors and apply negative buffer to reduce touching instances
     - Grid imagery/labels to a predefined grid (master grid) or to an adjustable square grid dimension of fixed height x width
         - apply usable data threshold on imagery
     - apply polygonization (if taking from consensus heatmap), morphology operations, connected components on labels
         - set instance masks to zero if they do not fall within min or max area threshold (used to remove spurious partial 
               masks and large ag if focusing only on smallholder) 
     - train/test split grid ids and move images/ instances to the following folder structure
         * train: subset of image chips for use in model training
                GRID_ID1
                    image: raw GeoTiff image chip from gridded imagery
                    masks: instance-specific GeoTiff masks for each object
           test: subset of image chips for use in model testing
                GRID_ID2
                    image: raw GeoTiff image chip from gridded imagery
                    masks: instance-specific GeoTiff masks for each object
     - finally, get array channel means, instance height/width and area distribution plot, and instance per image plot for 
           model_configs.py
    """
    
    def __init__(self, dataset_dir_name, images_source_dir, labels_source_dir, results_sub_dir_name, channel_indices, seasons):
        """ Set dataset name and other configs that depend on the specifc run but not the imagery or label processing."""
        
        self.ROOT_DIR = "../../data/"
        self.DATASET_DIR = os.path.join(self.ROOT_DIR, dataset_dir_name) #all processed data and outputs goes in here
        self.MASTER_GRID_PATH = os.path.join(self.ROOT_DIR, 'master_steph_grid.csv') # master grid over 8 South Africa wv2 scenes
        self.REORDERED_DIR = os.path.join(self.DATASET_DIR, 'reordered_tifs')
        self.TRAIN_DIR = os.path.join(self.DATASET_DIR, 'train')
        self.TEST_DIR = os.path.join(self.DATASET_DIR, 'test')
        self.WV2_GRID_DIR = os.path.join(self.DATASET_DIR, 'gridded_tifs')
        self.LABELS_GRID_DIR = os.path.join(self.DATASET_DIR, 'gridded_labels')
        self.CONNECTED_COMP_DIR = os.path.join(self.DATASET_DIR, 'connected_component_labels')
        self.OPENED_LABELS_DIR = os.path.join(self.DATASET_DIR, 'opened_labels')
        # Results directory
        # Save submission files and test/train split csvs here
        self.RESULTS_DIR = os.path.join(self.ROOT_DIR, "results/" + results_sub_dir_name) 

        self.IMAGES_SOURCE_DIR = os.path.join(self.ROOT_DIR, images_source_dir) #not gridded raw
        self.LABELS_SOURCE_DIR = os.path.join(self.ROOT_DIR, labels_source_dir) #not gridded raw
        
    # These will change with different configs
    
    
    SEASONS = ['GS']
    
    ADJUSTABLE_GRID_SIZE = 512
    
    USABLE_DATA_THRESHOLD = .25
    
    NEG_BUFFER = 2 # in meters, neg buffer on vectors before they are rasterized
    
    OPENING_KERNEL = (5, 5) # for seperating some touching fields into distinct fields, partialy deals with touching pixels.    
    

class WV2Config(PPConfig):
    """Configuration for preprocessing worldview-2 imagery. 
    
    Descriptive documentation for each attribute is at
    https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py"""
    
    def __init__(self):
        """
        
        """    

    
    # Image mean (RGBN RGBN) from WV2_MRCNN_PRE.ipynb
    # filling with N values, need to compute mean of each channel
    # values are for gridded wv2 no partial grids
    MEAN_PIXEL = np.array([200.05, 274.7, 164.04])
    
    # Give the configuration a recognizable name
    NAME = "wv2-512-cp-labels-allgrowing"

    # Use smaller anchors because our image and objects are small.
    # Setting Large upper scale since some fields take up nearly 
    # whole image
    RPN_ANCHOR_SCALES = (25, 75, 125, 250, 350)  # anchor side in pixels, determined using inspect_crop_data.ipynb
    
    #reduces the max number of field instances
    MAX_GT_INSTANCES = 63 # determined using inspect_crop_data.ipynb

    

