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
from difflib import SequenceMatcher
import yaml

import gdal # for gridding

random.seed(4)

def preprocess():
    
    def parse_yaml(input_file):
    """Parse yaml file of configuration parameters."""
    with open(input_file, 'r') as yaml_file:
        params = yaml.load(yaml_file)
    return params

    params = parse_yaml('preprocess_config.yaml') 

    ROOT = params['dirs']['root']

    DATASET = os.path.join(
        ROOT, params['dirs']['dataset'])

    REORDER = os.path.join(
        DATASET, params['dirs']['reorder'])

    TRAIN = os.path.join(
        DATASET, params['dirs']['train'])

    TEST = os.path.join(
        DATASET, params['dirs']['test'])

    GRIDDED_IMGS = os.path.join(
        DATASET, params['dirs']['gridded_imgs'])

    GRIDDED_LABELS = os.path.join(
        DATASET, params['dirs']['gridded_labels'])

    OPENED = os.path.join(
        DATASET, params['dirs']['opened'])

    INSTANCES = os.path.join(
        DATASET, params['dirs']['instances'])
    
    NEG_BUFFERED = os.path.join(
        DATASET, params['dirs']['neg_buffered_labels'])

    RESULTS = os.path.join(ROOT,'../',params['dirs']['results'], params['dirs']['dataset'])

    SOURCE_IMGS = os.path.join(
        ROOT, params['dirs']['source_imgs'])

    SOURCE_LABELS = os.path.join(
        ROOT, params['dirs']['source_labels'])

    dirs = [DATASET, REORDER, TRAIN, TEST, GRIDDED_IMGS, GRIDDED_LABELS, OPENED, INSTANCES, RESULTS]

    # Make directory and subdirectories
    for d in dirs:
        pathlib.Path(d).mkdir(parents=False, exist_ok=False)

    # Change working directory to project directory
    os.chdir(dirs[1])
    
    def remove_dir_folders(directory):
        '''
        Removes all files and sub-folders in a folder and keeps the folder.
        '''
    
        folderlist = [ f for f in os.listdir(directory)]
        for f in folderlist:
            shutil.rmtree(os.path.join(directory,f))
    
    def yaml_to_band_index(params):
        band_list = []
        for i, band in enumerate(params['bands_to_include']):
            if list(band.values())[0]== True:
                band_list.append(i)
        return band_list
    
    def reorder_images(params):
    """Load the os, gs, or both images and subset bands. Growing
    Season is stacked first before OS if both true.
    """
        file_ids_all = next(os.walk(SOURCE_IMGS))[2]
        band_indices = yaml_to_band_index(params)
        image_ids_gs = sorted([image_id for image_id in file_ids_all \
                               if 'GS' in image_id and '.aux' not in image_id])
        image_ids_os = sorted([image_id for image_id in file_ids_all \
                               if 'OS' in image_id and '.aux' not in image_id])

        if params['seasons']['GS'] and params['seasons']['OS'] == False:
            for img_path in image_ids_gs:
                gs_image = skio.imread(os.path.join(SOURCE_IMGS, img_path))
                gs_image = gs_image[:,:,band_indices]
                skio.imsave(gs_image_path, gs_image, plugin='tifffile')

        elif params['seasons']['OS'] and params['seasons']['GS'] == False:
            for img_path in image_ids_os:
                os_image = skio.imread(os.path.join(SOURCE_IMGS, img_path))
                os_image = gs_image[:,:,band_indices]
                skio.imsave(os_image_path, os_image, plugin='tifffile')
        else:
            for gs_path, os_path in zip(image_ids_gs, image_ids_os):
                print(gs_path, os_path)
                gs_image = skio.imread(os.path.join(SOURCE_IMGS, gs_path))
                os_image = skio.imread(os.path.join(SOURCE_IMGS, os_path))
                gsos_image = np.dstack([gs_image[:,:,band_indices], os_image[:,:,band_indices]])

                match = SequenceMatcher(None, gs_path, os_path).find_longest_match(0, len(gs_path), 0, len(os_path))
                path = gs_path[match.b: match.b + match.size] 
                # this may need to be reworked for diff file names
                # works best if unique ids like GS go in front of filename
                gsos_image_path = os.path.join(REORDER, path + 'OSGS.tif')
                skio.imsave(gsos_image_path, gsos_image, plugin='tifffile')
                
    def negative_buffer(params):
        """
        Applies a negative buffer to wv2 labels since they are too clsoe together and 
        produce conjoined instances when connected components is performed (even after 
        erosion/dilation). This may not get rid of all conjoinments and should be adjusted.
        It relies too on the source projection of the label file to calculate distances for
        the negative buffer. Unsure at what scale projection would matter in calculating this 
        distance.
        """
        neg_buffer = float(params['label_vals']['neg_buffer'])
        # This is a helper  used with sorted for a list of strings by specific indices in 
        # each string. Was used for a long path that ended with a file name
        # Not needed here but may be with different source imagery and labels
        def takefirst_two(elem):
            return int(elem[-12:-10])

        items = os.listdir(SOURCE_LABELS)
        labels = []
        for name in items:
            if name.endswith(".shp"):
                labels.append(os.path.join(SOURCE_LABELS,name))  

        shp_list = sorted(labels)

        scenes = os.listdir(SOURCE_IMGS)

        img_list = []
        for name in scenes:
            img_list.append(os.path.join(SOURCE_IMGS,name))  

        img_list = sorted(img_list)


        for shp_path, img_path in zip(shp_list, img_list):
            print(shp_path)
            shp_frame = gpd.read_file(shp_path)

            with rasterio.open(img_path) as rast:
                meta = rast.meta.copy()
                meta.update(compress="lzw")
                meta['count'] = 1

            rasterized_name = os.path.join(NEG_BUFFERED, os.path.basename(shp_path))
            with rasterio.open(rasterized_name, 'w+', **meta) as out:
                out_arr = out.read(1)
                # this is where we create a generator of geom, value pairs to use in rasterizing
                shp_frame['DN'].iloc[0] = 0
                shp_frame['DN'].iloc[1:] = 1
                maxx_bound = shp_frame.bounds.maxx.max()
                minx_bound = shp_frame.bounds.minx.min()
                if maxx_bound >= 30 and minx_bound>= 30:
                    shp_frame = shp_frame.to_crs({'init': 'epsg:32736'})
                    shp_frame['geometry'] = shp_frame['geometry'].buffer(neg_buffer)
                    shp_frame = shp_frame.to_crs({'init': 'epsg:4326'})

                else:
                    shp_frame = shp_frame.to_crs({'init': 'epsg:32735'})
                    shp_frame['geometry'] = shp_frame['geometry'].buffer(neg_buffer)
                    shp_frame = shp_frame.to_crs({'init': 'epsg:4326'})

                # hacky way of getting rid of empty geometries
                shp_frame = shp_frame[shp_frame.Shape_Area > 9e-11]
                shapes = ((geom,value) for geom, value in zip(shp_frame.geometry, shp_frame.DN))

                burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
                out.write_band(1, burned)
                
    
                
    def grid_images(params):
        """
        Grids up imagery to a variable size. Filters out imagery with too little usable data.
        """
        
        reorder_images(params)
        usable_data_threshold = params['image_vals']['usable_thresh']
        label_list = sorted(next(os.walk(SOURCE_LABELS))[2])
        img_list = sorted(next(os.walk(SOURCE_LABELS))[2])
        
        
    
    def remove_dir_folders(directory):
        '''
        Removes all files and sub-folders in a folder and keeps the folder.
        '''
    
        folderlist = [ f for f in os.listdir(directory)]
        for f in folderlist:
            shutil.rmtree(os.path.join(directory,f))

    image_list = next(os.walk(REORDERED_DIR))[2]
    
    def move_img_to_folder(filename):
        '''Moves a file with identifier pattern ZA0165086_MS_GS.tif to a 
        folder path ZA0165086/image/ZA0165086_MS_GS.tif
        Also creates a masks folder at ZA0165086/masks'''
        
        folder_name = os.path.join(TRAIN_DIR,filename[:9])
        if os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.mkdir(folder_name)
        new_path = os.path.join(folder_name, 'image')
        mask_path = os.path.join(folder_name, 'masks')
        os.mkdir(new_path)
        file_path = os.path.join(REORDERED_DIR,filename)
        os.rename(file_path, os.path.join(new_path, filename))
        os.mkdir(mask_path)

    for img in image_list:
        move_img_to_folder(img)

    label_list = next(os.walk(LABELS_DIR))[2]

    for name in label_list:
        arr = skio.imread(os.path.join(LABELS_DIR,name))
        arr[arr == -1.7e+308]=0
        label_name = name[0:15]+'.tif'
        opened_path = os.path.join(OPENED_LABELS_DIR,name)
        kernel = np.ones((5,5))
        arr = skim.binary_opening(arr, kernel)
        arr=1*arr
        assert arr.ndim == 2
        assert arr.shape == (512, 512)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            skio.imsave(opened_path, 1*arr)

    label_list = next(os.walk(OPENED_LABELS_DIR))[2]
    # save connected components and give each a number at end of id
    for name in label_list:
        arr = skio.imread(os.path.join(OPENED_LABELS_DIR,name))
        blob_labels = measure.label(arr, background=0)
        blob_vals = np.unique(blob_labels)
        for blob_val in blob_vals[blob_vals!=0]:
            labels_copy = blob_labels.copy()
            assert labels_copy.shape == (512, 512)
            labels_copy[blob_labels!=blob_val] = 0
            labels_copy[blob_labels==blob_val] = 1
            label_name = name[0:15]+str(blob_val)+'.tif'
            label_path = os.path.join(CONNECTED_COMP_DIR,label_name)
            assert labels_copy.ndim == 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                skio.imsave(label_path, labels_copy)

    def move_mask_to_folder(filename):
        '''Moves a mask with identifier pattern ZA0165086_label_1.tif to a 
        folder path ZA0165086/mask/ZA0165086_label_1.tif. Need to run 
        connected components first.
        '''
        if os.path.isdir(os.path.join(TRAIN_DIR,filename[:9])):
            folder_path = os.path.join(TRAIN_DIR,filename[:9])
            mask_path = os.path.join(folder_path, 'masks')
            file_path = os.path.join(CONNECTED_COMP_DIR,filename)
            os.rename(file_path, os.path.join(mask_path, filename))

    mask_list = next(os.walk(CONNECTED_COMP_DIR))[2]
    for mask in mask_list:
        move_mask_to_folder(mask)

    id_list = next(os.walk(TRAIN_DIR))[1]
    
    for fid in id_list:
        mask_folder = os.path.join(DATASET_DIR, 'train',fid, 'masks')
        im_folder = os.path.join(DATASET_DIR, 'train',fid, 'image')
        if not os.listdir(mask_folder):
            im_path = os.path.join(im_folder, os.listdir(im_folder)[0])
            arr = skio.imread(im_path)
            assert arr.shape == (512, 512, 3)
            mask = np.zeros_like(arr[:,:,0])
            assert mask.shape == (512, 512)
            assert mask.ndim == 2
            # ignores warning about low contrast image
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                skio.imsave(os.path.join(mask_folder, fid + '_label_0.tif'),mask)

    def train_test_split(train_dir, test_dir, kprop):
        """Takes a sample of folder ids and copies them to a test directory
        from a directory with all folder ids. Each sample folder contains an 
        images and corresponding masks folder."""

        remove_dir_folders(test_dir)
        sample_list = next(os.walk(train_dir))[1]
        k = round(kprop*len(sample_list))
        test_list = random.sample(sample_list,k)
        for test_sample in test_list:
            shutil.copytree(os.path.join(train_dir,test_sample),os.path.join(test_dir,test_sample))
        train_list = list(set(next(os.walk(train_dir))[1]) - set(test_list))
        train_df = pd.DataFrame({'train': train_list})
        test_df = pd.DataFrame({'test': test_list})
        train_df.to_csv(os.path.join(RESULTS_DIR, 'train_ids.csv'))
        test_df.to_csv(os.path.join(RESULTS_DIR, 'test_ids.csv'))
        
    train_test_split(TRAIN_DIR, TEST_DIR, .1)
    print('preprocessing complete, ready to run model.')

def get_arr_channel_mean(channel):
    means = []
    for i, fid in enumerate(id_list):
        im_folder = os.path.join('train',fid, 'image')
        im_path = os.path.join(im_folder, os.listdir(im_folder)[0])
        arr = skio.imread(im_path)
        arr[arr==-1.7e+308]=np.nan
        means.append(np.nanmean(arr[:,:,channel]))
    return np.mean(means)
