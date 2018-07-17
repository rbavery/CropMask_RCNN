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
    
    def __init__(self, dataset_dir_name, images_source_dir, labels_source_dir):
        """ Set dataset name and other configs that depend on the specifc run but not the imagery or label processing."""
        
        self.DATASET_DIR = os.path.join(self.ROOT_DIR, dataset_dir_name)#all processed data and outputs goes in here
        self.IMAGES_SOURCE_DIR = os.path.join(self.ROOT_DIR, image_source_dir) #not gridded raw
        self.LABELS_SOURCE_DIR = os.path.join(self.ROOT_DIR, labels_source_dir) #not gridded raw
        
        
    # Paths that shouldn't change with different imagery sources
    ROOT_DIR = "../../data/"
    MASTER_GRID_PATH = os.path.join(ROOT_DIR, 'master_steph_grid.csv') # master grid over 8 South Africa wv2 scenes
    REORDERED_DIR = os.path.join(DATASET_DIR, 'reordered_tifs')
    TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
    TEST_DIR = os.path.join(DATASET_DIR, 'test')
    WV2_GRID_DIR = os.path.join(DATASET_DIR, 'gridded_tifs')
    LABELS_GRID_DIR = os.path.join(DATASET_DIR, 'gridded_labels')
    CONNECTED_COMP_DIR = os.path.join(DATASET_DIR, 'connected_component_labels')
    OPENED_LABELS_DIR = os.path.join(DATASET_DIR, 'opened_labels')
    # Results directory
    # Save submission files and test/train split csvs here
    RESULTS_DIR = os.path.join(ROOT_DIR, "results/wv2/") 

    # These will change with different configs
    
    CHANNEL_INDICES = [4, 2, 1] #default for wv2 RGB
    
    SEASONS = ['GS']
    
    ADJUSTABLE_GRID_SIZE = 512
    
    USABLE_DATA_THRESHOLD = .25
    
    NEG_BUFFER = 2 # in meters, neg buffer on vectors before they are rasterized
    
    OPENING_KERNEL = (5, 5) # for seperating some touching fields into distinct fields, partialy deals with touching pixels.    
    

class WV2Config():
    """Configuration for preprocessing worldview-2 imagery. 
    
    Descriptive documentation for each attribute is at
    https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/config.py"""
    
    def __init__(self, N):
        """Set values of computed attributes. Channel dimension is overriden, 
        replaced 3 with N as per this guideline: https://github.com/matterport/Mask_RCNN/issues/314
        THERE MAY BE OTHER CODE CHANGES TO ACCOUNT FOR 3 vs N channels. See other 
        comments."""
        # https://github.com/matterport/Mask_RCNN/wiki helpful for N channels
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        
        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, N])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, N])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
        self.CHANNELS_NUM = N
    
    LEARNING_RATE = .0001 
    
    # Image mean (RGBN RGBN) from WV2_MRCNN_PRE.ipynb
    # filling with N values, need to compute mean of each channel
    # values are for gridded wv2 no partial grids
    MEAN_PIXEL = np.array([200.05, 274.7, 164.04])
    
    # Give the configuration a recognizable name
    NAME = "wv2-512-cp-labels-allgrowing"

    # Batch size is 4 (GPUs * images/GPU).
    # New parralel_model.py allows for multi-gpu
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + ag

    # Use small images for faster training. Determines the image shape.
    # From build() in model.py
    # Exception("Image size must be dividable by 2 at least 6 times "
    #     "to avoid fractions when downscaling and upscaling."
    #    "For example, use 256, 320, 384, 448, 512, ... etc. "
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small.
    # Setting Large upper scale since some fields take up nearly 
    # whole image
    RPN_ANCHOR_SCALES = (25, 75, 125, 250, 350)  # anchor side in pixels, determined using inspect_crop_data.ipynb

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 500
    
    #reduces the max number of field instances
    MAX_GT_INSTANCES = 63 # determined using inspect_crop_data.ipynb

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 100
    
    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "resnet50"
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    


class WV2InferenceConfig(WV2Config):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imagery for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
