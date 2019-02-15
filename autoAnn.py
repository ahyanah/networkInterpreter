#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os, sys
import json
import time
import skimage.io
import configparser
import numpy as np
import tensorflow as tf


# In[2]:


def getSectionToUse(config_dict):
    ini_section = 'DEFAULT'
    if config_dict.has_option('DEFAULT', 'section_to_use'):
        if config_dict['DEFAULT']['section_to_use'] != '':
            ini_section = config_dict['DEFAULT']['section_to_use']
            if ini_section not in config_dict.sections():
                raise ValueError('section_to_use  ' + ini_section + ' stated in env.ini is not a valid section.')
    else:
        print('Section_to_use not specified in env.ini. Using Default values.')
    return ini_section

# In[3]:


config_dict = configparser.ConfigParser()
config_dict.read('env.ini')
ini_section = getSectionToUse(config_dict)
RCNN_DIR = config_dict[ini_section]['RCNN_DIR']
if not os.path.isdir(RCNN_DIR):
    raise NotADirectoryError(RCNN_DIR + " in env.ini is not a directory.")


# In[4]:


# SET UP RCNN LOGS AND COCO DEPENDENCIES 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(RCNN_DIR, "logs")
print('Creating model')
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(RCNN_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
sys.path.append(os.path.abspath(RCNN_DIR))
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(RCNN_DIR, "samples/coco/"))  # To find local version
import coco


# In[5]:


def detectColour(image, aoi, threshold=0.3 ):
    # define the list of boundaries
    boundaries = [ #RGB
        ([100, 50, 50], [255, 130, 130], 'RED'),
        ([50, 100, 50], [130, 255, 130], 'GREEN'),
        ([50, 50, 100], [130, 130, 255], 'BLUE'),
        ([100, 100, 50], [255, 255, 130], 'YELLOW' ),
        ([100, 100, 100], [225, 225, 225], 'SILVER' ), 
        ([225, 225, 225], [255, 255, 255], 'WHITE' ),
        ([0, 0, 0], [100, 100, 100], 'BLACK' ) 
    ]
    # loop over the boundaries
    for (lower, upper, colour) in boundaries:
    # (lower, upper, colour) = boundaries[4]
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)

        tmp_mask = np.logical_and(mask, aoi)
        output = cv2.bitwise_and(image, image, mask = tmp_mask.astype("uint8"))
        # print(np.sum(np.logical_and(mask, aoi))/np.sum(aoi))
        isColour = np.sum(np.logical_and(mask, aoi))/np.sum(aoi) > threshold
        if isColour:
            return colour, output
            
    return "OTHERS", output


# In[6]:


def getImageIdfrmPath(path, case='oneMotoring'):
    image_id = ''
    # for onemotoring:
    if case.lower() == 'oneMotoring'.lower():
        path, im = os.path.split(path)
        path, date = os.path.split(path)
        path, location = os.path.split(path)
        image_id = location.replace(' ', '_') + '-' + date + '-' + im.replace('.jpeg','')
    return image_id


# In[7]:


def initializeVariables(config_dict):
    ini_section = getSectionToUse(config_dict)
    # Initialise of output<dict> with items in reference.json
    F_ref = open(config_dict[ini_section]['reference_file'],'r')  # TODO: add error checks
    text_ = F_ref.read()
    json_ref = json.loads(text_)
    jsonOutput = {'categories': json_ref['categories'],                  'annotations': [],                  'images': json_ref['images'],                  'licenses': json_ref['licenses']}
    F_ref.close()
    
    if not config_dict.has_option(ini_section, 'obj_of_interest'):
        raise ValueError('Object of interest not found in env.ini')
    
    if not config_dict.has_option(ini_section, 'im_dir'):
        raise ValueError('Image directory not found in env.ini')
        
    if not config_dict.has_option(ini_section, 'output_file'):
        raise ValueError('Output_file path not found in env.ini')
    
    if not config_dict.has_option(ini_section, 'batch_size'):
        raise ValueError('batch_size not found in env.ini')
    
    if not config_dict.has_option(ini_section, 'threshold'):
        raise ValueError('Min. size for object detection, threshold, not found in env.ini')
        
    if not config_dict.has_option(ini_section, 'ann_id_prefix'):
        raise ValueError('Annotation id prefix, ann_id_prefix not found in env.ini')
    return jsonOutput, config_dict[ini_section]

def get_lines_as_list(file_path):
    File = open(file_path, 'r')
    lines = File.readlines()
    lines_in_list = [x.strip() for x in lines] 
    File.close()
    return lines_in_list


# In[8]:


# INITIALISATION OF VARIABLES
jsonOutput, config_ = initializeVariables(config_dict)  # use config with underscore to prevent conflict with pyco's config
OBJECTS_OF_INTEREST_ls = config_['obj_of_interest']
IM_DIR = config_['im_dir']
output_file = config_['output_file']
BATCH_SIZE = int(config_['batch_size'])
THRESHOLD = float(config_['threshold'])
ANN_ID_PREFIX = config_['ann_id_prefix']
OBJECTS_OF_INTEREST_ls = [ int(x) for x in OBJECTS_OF_INTEREST_ls.replace('[','').replace(']','').split(',')]
done_list_path = config_['done_list']
'BUS' in ', '.join([items['id'] for items in jsonOutput['categories']]).split(', ')
ann_json_list = []
if len(done_list_path) != 0:
    done_list = get_lines_as_list(done_list_path)
else:
    done_list = []
global_counter = 1
other_list = []


# In[9]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign']

# OVERRIDES MASK_RCNN DEFAULTS
class_names = ['BG', 'person', 'bicycle', 'SEDAN', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign']

# REMOVES OBJECTS THAT ARE OF INTEREST BUT ARE NOT IN REFERENCE CATEGORY LIST
categories_list = ', '.join([items['id'] for items in jsonOutput['categories']]).lower().split(', ')
ref_indices_to_be_removed = []
for obj_ref_idx in OBJECTS_OF_INTEREST_ls:
    if not class_names[obj_ref_idx].lower() in categories_list:
        ref_indices_to_be_removed.append(obj_ref_idx)
        print(class_names[obj_ref_idx])
for i in ref_indices_to_be_removed:
    print('Removing ' + class_names[i] + ' from objects-of-interest list as it is not present in the category list of reference.json')
    OBJECTS_OF_INTEREST_ls.remove(i)


# In[10]:


# Directory to save logs and trained model
MODEL_DIR = os.path.join(RCNN_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(RCNN_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = BATCH_SIZE

inf_config = InferenceConfig()
inf_config.display()


# In[ ]:

list_of_image_paths = []
assert os.path.isdir(IM_DIR), IM_DIR + " is an invalid image directory."

for path, subdirs, files in os.walk(IM_DIR):
    for file in files:
        if file.endswith('.jpeg'):
            list_of_image_paths.append(os.path.join(path, file))

# MAIN
# Create model object in inference mode.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=inf_config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # MAIN
    # TODO track done images and images with objects not of interest


    print('Found ' + str(len(list_of_image_paths)) + ' images.')
    im_generator = enumerate(list_of_image_paths)
    global_counter = 0
    end_of_list = False
    start = time.time() 
    if (len(list_of_image_paths)==0):
        print('No images was found in ' + IM_DIR)
        exit()

    while not end_of_list:
        image_array = []
        image_sz_array = []
        image_path_array = []
        for i in range(BATCH_SIZE):
            try: im_counter, nxt_im_path = im_generator.__next__()
            except StopIteration: end_of_list = True; break
            image = skimage.io.imread(nxt_im_path)
            image_array.append(image)
            image_sz_array.append(image.shape)  # height, width, _ 
            image_path_array.append(nxt_im_path)

        if len(image_array) == BATCH_SIZE:
            test_results = model.detect(image_array, verbose=0)
        else:
            # initialise a model with the corresponding BATCH_SIZE
            config_new_config = InferenceConfig()
            config_new_config.BATCH_SIZE = len(image_array)
            config_new_config.IMAGES_PER_GPU = len(image_array)
            config_new_config.NAME = "coco_odd_size"
            config_new_config.display()
            # Create model object in inference mode.
            model_odd_batch_size = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config_new_config)
            # Load weights trained on MS-COCO
            model_odd_batch_size.load_weights(COCO_MODEL_PATH, by_name=True)
            test_results = model_odd_batch_size.detect(image_array, verbose=0)

        for ii in range(len(test_results)):
            r = test_results[ii]
            image_id = getImageIdfrmPath(image_path_array[ii])
            height, width, _ = image_sz_array[ii]
            for idx, roi in enumerate(r['rois']):
                if len(OBJECTS_OF_INTEREST_ls) != 0 and r['class_ids'][idx] not in OBJECTS_OF_INTEREST_ls:
                    other_list.append({'category_id': idx, "image_id": image_id})
                    continue
                y1, x1, y2, x2 = roi
                aoi = r['masks'][y1:y2, x1:x2,idx]
                input = image_array[ii][y1:y2, x1:x2]
                colour, _ = detectColour(input, aoi)

                bbox = [float(x1)/width, float(y1)/height, float(x2-x1)/width, float(y2-y1)/height]  # Normalise for annotation tool
                if ((bbox[2]<THRESHOLD) or (bbox[3]<THRESHOLD)):
                    continue
                cat_id = class_names[r['class_ids'][idx]].upper()

                if cat_id != 'MOTORCYCLE':
                    ann = {"id" : ANN_ID_PREFIX+str(global_counter), "image_id": image_id, 'category_id': cat_id +", "+colour, 'bbox': bbox}
                else:
                    ann = {"id" : ANN_ID_PREFIX+str(global_counter), "image_id": image_id, 'category_id': cat_id, 'bbox': bbox}

                global_counter+=1
                ann_json_list.append(ann)
            if len(r['rois'])==0:
                print('Nothing detected in image with id: '+ image_id)


        if ((im_counter != 0) and ((im_counter+1) % 100)==0):
            print(str(im_counter+1) + ' image(s) done. ' + str(len(list_of_image_paths)-im_counter-1) + ' more to go. Hang in there!')
            with open(output_file.replace('.json','_done_list_' + str(im_counter+1) + '.txt'), 'w') as f:
                for item in list_of_image_paths:
                    f.write("%s\n" % item)

            if len(ann_json_list)==0:
                print('Nothing detected yet.')
            else:
                jsonOutput['annotations'] = ann_json_list
                File = open(output_file.replace('.json', '_'+ str(im_counter) + '.json'),'w')
                File.write(json.dumps(jsonOutput))
                File.close()
    end = time.time()
    print(end - start)

    if len(ann_json_list)==0:
        print('Nothing detected.')
    else:
        jsonOutput['annotations'] = ann_json_list
        File = open(output_file,'w')
        File.write(json.dumps(jsonOutput))
        File.close()


    with open(output_file.replace('.json','_done_list.txt'), 'w') as f:
        for item in list_of_image_paths:
            f.write("%s\n" % item)

    if (len(other_list) != 0):
        File = open(output_file.replace('.json','_strange_list.json'),'w')
        File.write(json.dumps({'strange objects': other_list}))
        File.close()

    print('Done.')
