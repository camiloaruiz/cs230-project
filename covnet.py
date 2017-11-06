# import libraries
from __future__ import print_function
from __future__ import division

import numpy as np 
import pandas as pd
import os
import re

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import random
from timeit import default_timer as timer
# imports
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import tsahelper as tsa
#----------------------------------------------------------------------------------
# read_header(infile):  takes an aps file and creates a dict of the data
#
# infile:               an aps file
#
# returns:              all of the fields in the header
#----------------------------------------------------------------------------------
#import tsahelper as tsa
#---------------------------------------------------------------------------------------
# Constants
#
# INPUT_FOLDER:                 The folder that contains the source data
#
# PREPROCESSED_DATA_FOLDER:     The folder that contains preprocessed .npy files 
# 
# STAGE1_LABELS:                The CSV file containing the labels by subject
#
# THREAT_ZONE:                  Threat Zone to train on (actual number not 0 based)
#
# BATCH_SIZE:                   Number of Subjects per batch
#
# EXAMPLES_PER_SUBJECT          Number of examples generated per subject
#
# FILE_LIST:                    A list of the preprocessed .npy files to batch
# 
# TRAIN_TEST_SPLIT_RATIO:       Ratio to split the FILE_LIST between train and test
#
# TRAIN_SET_FILE_LIST:          The list of .npy files to be used for training
#
# TEST_SET_FILE_LIST:           The list of .npy files to be used for testing
#
# IMAGE_DIM:                    The height and width of the images in pixels
#
# LEARNING_RATE                 Learning rate for the neural network
#
# N_TRAIN_STEPS                 The number of train steps (epochs) to run
#
# TRAIN_PATH                    Place to store the tensorboard logs
#
# MODEL_PATH                    Path where model files are stored
#
# MODEL_NAME                    Name of the model files
#
#----------------------------------------------------------------------------------------
INPUT_FOLDER = 'tsa_datasets/stage1/aps/stage1_aps'
PREPROCESSED_DATA_FOLDER = 'tsa_datasets/preprocessed/'
STAGE1_LABELS = 'tsa_datasets/stage1_labels.csv'
THREAT_ZONE = 1
BATCH_SIZE = 16
EXAMPLES_PER_SUBJECT = 182

FILE_LIST = []
TRAIN_TEST_SPLIT_RATIO = 0.2
TRAIN_SET_FILE_LIST = []
TEST_SET_FILE_LIST = []

IMAGE_DIM = 250
LEARNING_RATE = 1e-3
N_TRAIN_STEPS = 1
TRAIN_PATH = 'tsa_logs/train/'
MODEL_PATH = 'tsa_logs/model/'
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM, IMAGE_DIM, THREAT_ZONE )) 
# constants
COLORMAP = 'pink'
APS_FILE_NAME = 'tsa_datasets/stage1/aps/00360f79fd6e02781457eda48f85da90.aps'
BODY_ZONES = 'tsa_datasets/stage1/body_zones.png'
THREAT_LABELS = 'tsa_datasets/stage1/stage1_labels.csv'

def read_header(infile):
    # declare dictionary
    h = dict()
    
    with open(infile, 'r+b') as fid:

        h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
        h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
        h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
        h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
        h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
        h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
        h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
        h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
        h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
        h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)

    return h
  
#unit test ----------------------------------
#header = read_header(APS_FILE_NAME)

#for data_item in sorted(header):
#    print ('{} -> {}'.format(data_item, header[data_item]))

#----------------------------------------------------------------------------------
# read_data(infile):  reads and rescales any of the four image types
#
# infile:             an .aps, .aps3d, .a3d, or ahi file
#
# returns:            the stack of images
#
# note:               word_type == 7 is an np.float32, word_type == 4 is np.uint16      
#----------------------------------------------------------------------------------

def read_data(infile):
    
    print ("getting here")
    # read in header and get dimensions
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    
    extension = os.path.splitext(infile)[1]
    
    with open(infile, 'rb') as fid:
          
        # skip the header
        fid.seek(512) 

        # handle .aps and .a3aps files
        if extension == '.aps' or extension == '.a3daps':
        
            if(h['word_type']==7):
                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)

            elif(h['word_type']==4): 
                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor'] 
            data = data.reshape(nx, ny, nt, order='F').copy()

        # handle .a3d files
        elif extension == '.a3d':
              
            if(h['word_type']==7):
                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
                
            elif(h['word_type']==4):
                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor']
            data = data.reshape(nx, nt, ny, order='F').copy() 
            
        # handle .ahi files
        elif extension == '.ahi':
            data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
            data = data.reshape(2, ny, nx, nt, order='F').copy()
            real = data[0,:,:,:].copy()
            imag = data[1,:,:,:].copy()

        if extension != '.ahi':
            return data
        else:
            return real, imag

#unit test ----------------------------------
#d = read_data(APS_FILE_NAME)
#----------------------------------------------------------------------------------
# plot_image_set(infile):  takes an aps file and shows all 16 90 degree shots
#
# infile:                  an aps file
#----------------------------------------------------------------------------------
# def plot_image_set(infile):

#     # read in the aps file, it comes in as shape(512, 620, 16)
#     img = read_data(infile)
    
#     # transpose so that the slice is the first dimension shape(16, 620, 512)
#     img = img.transpose()
        
#     # show the graphs
#     fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
#     fig.savefig(infile + ".png")
    
#     i = 0
#     resized_img = cv2.resize(img[0], (0,0), fx=0.1, fy=0.1)
#     plt.imshow(resized_img)
#     plt.close90
"""
    for row in range(4):
        for col in range(4):
            resized_img = cv2.resize(img[i], (0,0), fx=0.1, fy=0.1)
            plt.imshow(resized_img)
            axarr[row, col].imshow(np.flipud(resized_img), cmap=COLORMAP)
            i += 1
    
    print('Done!')
"""

#----------------------------------------------------------------------------------
# plot_image_set(infile):  takes an aps file and shows all 16 90 degree shots
#
# infile:                  an aps file
#----------------------------------------------------------------------------------
# def plot_image_set(infile):

#     # read in the aps file, it comes in as shape(512, 620, 16)
#     img = read_data(infile)
    
#     # transpose so that the slice is the first dimension shape(16, 620, 512)
#     img = img.transpose()
        
#     # show the graphs
#     fig, axarr = plt.subplots(nrows=4, ncols=4, figsize=(10,10))
    
#     i = 0
#     for row in range(4):
#         for col in range(4):
#             resized_img = cv2.resize(img[i], (0,0), fx=0.1, fy=0.1)
#             axarr[row, col].imshow(np.flipud(resized_img), cmap=COLORMAP)
#             i += 1
#     fig.savefig(infile + ".png")
#     print('Done!')

# unit test ----------------------------------
#plot_image_set(APS_FILE_NAME)

# unit test ----------------------------------
#plot_image_set(APS_FILE_NAME)
# plot_image_set("00360f79fd6e02781457eda48f85da90.aps")
#plot_image_set("00360f79fd6e02781457eda48f85da90.ahi")


#---------------------------------------------------------------------------------------
# preprocess_tsa_data(): preprocesses the tsa datasets
#
# parameters:      none
#
# returns:         none
#---------------------------------------------------------------------------------------

def preprocess_tsa_data():
    
    # OPTION 1: get a list of all subjects for which there are labels
    #df = pd.read_csv(STAGE1_LABELS)
    #df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    #SUBJECT_LIST = df['Subject'].unique()

    # OPTION 2: get a list of all subjects for whom there is data
    #SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(INPUT_FOLDER)]
    
    # OPTION 3: get a list of subjects for small bore test purposes
    SUBJECT_LIST = ['00360f79fd6e02781457eda48f85da90','0043db5e8c819bffc15261b1f1ac5e42',
                    '0050492f92e22eed3474ae3a6fc907fa','006ec59fa59dd80a64c85347eef810c7',
                    '0097503ee9fa0606559c56458b281a08','011516ab0eca7cad7f5257672ddde70e']
    
    # intialize tracking and saving items
    batch_num = 1
    threat_zone_examples = []
    start_time = timer()
    
    for subject in SUBJECT_LIST:

        # read in the images
        print('--------------------------------------------------------------')
        print('t+> {:5.3f} |Reading images for subject #: {}'.format(timer()-start_time, 
                                                                     subject))
        print('--------------------------------------------------------------')
        images = read_data(INPUT_FOLDER + '/' + subject + '.aps')

        # transpose so that the slice is the first dimension shape(16, 620, 512)
        images = images.transpose()

        # for each threat zone, loop through each image, mask off the zone and then crop it
        for tz_num, threat_zone_x_crop_dims in enumerate(zip(tsa.zone_slice_list, 
                                                             tsa.zone_crop_list)):

            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            # get label
            label = np.array(tsa.get_subject_zone_label(tz_num, 
                             tsa.get_subject_labels(STAGE1_LABELS, subject)))

            for img_num, img in enumerate(images):

                print('Threat Zone:Image -> {}:{}'.format(tz_num, img_num))
                print('Threat Zone Label -> {}'.format(label))
                
                if threat_zone[img_num] is not None:

                    # correct the orientation of the image
                    print('-> reorienting base image') 
                    base_img = np.flipud(img)
                    print('-> shape {}|mean={}'.format(base_img.shape, 
                                                       base_img.mean()))

                    # convert to grayscale
                    print('-> converting to grayscale')
                    rescaled_img = tsa.convert_to_grayscale(base_img)
                    print('-> shape {}|mean={}'.format(rescaled_img.shape, 
                                                       rescaled_img.mean()))

                    # spread the spectrum to improve contrast
                    print('-> spreading spectrum')
                    high_contrast_img = tsa.spread_spectrum(rescaled_img)
                    print('-> shape {}|mean={}'.format(high_contrast_img.shape,
                                                       high_contrast_img.mean()))

                    # get the masked image
                    print('-> masking image')
                    masked_img = tsa.roi(high_contrast_img, threat_zone[img_num])
                    print('-> shape {}|mean={}'.format(masked_img.shape, 
                                                       masked_img.mean()))

                    # crop the image
                    print('-> cropping image')
                    cropped_img = tsa.crop(masked_img, crop_dims[img_num])
                    print('-> shape {}|mean={}'.format(cropped_img.shape, 
                                                       cropped_img.mean()))

                    # normalize the image
                    print('-> normalizing image')
                    normalized_img = tsa.normalize(cropped_img)
                    print('-> shape {}|mean={}'.format(normalized_img.shape, 
                                                       normalized_img.mean()))

                    # zero center the image
                    print('-> zero centering')
                    zero_centered_img = tsa.zero_center(normalized_img)
                    print('-> shape {}|mean={}'.format(zero_centered_img.shape, 
                                                       zero_centered_img.mean()))

                    # append the features and labels to this threat zone's example array
                    print ('-> appending example to threat zone {}'.format(tz_num))
                    threat_zone_examples.append([[tz_num], zero_centered_img, label])
                    print ('-> shape {:d}:{:d}:{:d}:{:d}:{:d}:{:d}'.format(
                                                         len(threat_zone_examples),
                                                         len(threat_zone_examples[0]),
                                                         len(threat_zone_examples[0][0]),
                                                         len(threat_zone_examples[0][1][0]),
                                                         len(threat_zone_examples[0][1][1]),
                                                         len(threat_zone_examples[0][2])))
                else:
                    print('-> No view of tz:{} in img:{}. Skipping to next...'.format( 
                                tz_num, img_num))
                print('------------------------------------------------')

        # each subject gets EXAMPLES_PER_SUBJECT number of examples (182 to be exact, 
        # so this section just writes out the the data once there is a full minibatch 
        # complete.
        if ((len(threat_zone_examples) % (BATCH_SIZE * EXAMPLES_PER_SUBJECT)) == 0):
            for tz_num, tz in enumerate(tsa.zone_slice_list):

                tz_examples_to_save = []

                # write out the batch and reset
                print(' -> writing: ' + PREPROCESSED_DATA_FOLDER + 
                                        'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format( 
                                        tz_num+1,
                                        len(threat_zone_examples[0][1][0]),
                                        len(threat_zone_examples[0][1][1]), 
                                        batch_num))

                # get this tz's examples
                tz_examples = [example for example in threat_zone_examples if example[0] == 
                               [tz_num]]

                # drop unused columns
                tz_examples_to_save.append([[features_label[1], features_label[2]] 
                                            for features_label in tz_examples])

                # save batch.  Note that the trainer looks for tz{} where {} is a 
                # tz_num 1 based in the minibatch file to select which batches to 
                # use for training a given threat zone
                np.save(PREPROCESSED_DATA_FOLDER + 
                        'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                                                         len(threat_zone_examples[0][1][0]),
                                                         len(threat_zone_examples[0][1][1]), 
                                                         batch_num), 
                                                         tz_examples_to_save)
                del tz_examples_to_save

            #reset for next batch 
            del threat_zone_examples
            threat_zone_examples = []
            batch_num += 1
    
    # we may run out of subjects before we finish a batch, so we write out 
    # the last batch stub
    if (len(threat_zone_examples) > 0):
        for tz_num, tz in enumerate(tsa.zone_slice_list):

            tz_examples_to_save = []

            # write out the batch and reset
            print(' -> writing: ' + PREPROCESSED_DATA_FOLDER 
                    + 'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                      len(threat_zone_examples[0][1][0]),
                      len(threat_zone_examples[0][1][1]), 
                                                                                                                  batch_num))

            # get this tz's examples
            tz_examples = [example for example in threat_zone_examples if example[0] == 
                           [tz_num]]

            # drop unused columns
            tz_examples_to_save.append([[features_label[1], features_label[2]] 
                                        for features_label in tz_examples])

            #save batch
            np.save(PREPROCESSED_DATA_FOLDER + 
                    'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                                                     len(threat_zone_examples[0][1][0]),
                                                     len(threat_zone_examples[0][1][1]), 
                                                     batch_num), 
                                                     tz_examples_to_save)
# unit test ---------------------------------------
preprocess_tsa_data()

#---------------------------------------------------------------------------------------
# get_train_test_file_list(): gets the batch file list, splits between train and test
#
# parameters:      none
#
# returns:         none
#
#-------------------------------------------------------------------------------------

def get_train_test_file_list():
    
    global FILE_LIST
    global TRAIN_SET_FILE_LIST
    global TEST_SET_FILE_LIST

    if os.listdir(PREPROCESSED_DATA_FOLDER) == []:
        print ('No preprocessed data available.  Skipping preprocessed data setup..')
    else:
        FILE_LIST = [f for f in os.listdir(PREPROCESSED_DATA_FOLDER) 
                     if re.search(re.compile('-tz' + str(THREAT_ZONE) + '-'), f)]
        train_test_split = len(FILE_LIST) - \
                           max(int(len(FILE_LIST)*TRAIN_TEST_SPLIT_RATIO),1)
        TRAIN_SET_FILE_LIST = FILE_LIST[:train_test_split]
        TEST_SET_FILE_LIST = FILE_LIST[train_test_split:]
        print('Train/Test Split -> {} file(s) of {} used for testing'.format( 
              len(FILE_LIST) - train_test_split, len(FILE_LIST)))
        
# unit test ----------------------------
#get_train_test_file_list()
#print (

#---------------------------------------------------------------------------------------
# input_pipeline(filename, path): prepares a batch of features and labels for training
#
# parameters:      filename - the file to be batched into the model
#                  path - the folder where filename resides
#
# returns:         feature_batch - a batch of features to train or test on
#                  label_batch - a batch of labels related to the feature_batch
#
#---------------------------------------------------------------------------------------

def input_pipeline(filename, path):

    preprocessed_tz_scans = []
    feature_batch = []
    label_batch = []
    
    #Load a batch of preprocessed tz scans
    preprocessed_tz_scans = np.load(os.path.join(path, filename))
        
    #Shuffle to randomize for input into the model
    np.random.shuffle(preprocessed_tz_scans)
    
    # separate features and labels
    for example_list in preprocessed_tz_scans:
        for example in example_list:
            feature_batch.append(example[0])
            label_batch.append(example[1])
    
    feature_batch = np.asarray(feature_batch, dtype=np.float32)
    label_batch = np.asarray(label_batch, dtype=np.float32)
    
    return feature_batch, label_batch
  
# unit test ------------------------------------------------------------------------
#print ('Train Set -----------------------------')
#for f_in in TRAIN_SET_FILE_LIST:
#    feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
#    print (' -> features shape {}:{}:{}'.format(len(feature_batch), 
#                                                len(feature_batch[0]), 
#                                                len(feature_batch[0][0])))
#    print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))
    
#print ('Test Set -----------------------------')
#for f_in in TEST_SET_FILE_LIST:
#    feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
#    print (' -> features shape {}:{}:{}'.format(len(feature_batch), 
#                                                len(feature_batch[0]), 
#                                                len(feature_batch[0][0])))
#    print (' -> labels shape   {}:{}'.format(len(label_batch), len(label_batch[0])))


#---------------------------------------------------------------------------------------
# shuffle_train_set(): shuffle the list of batch files so that each train step
#                      receives them in a different order since the TRAIN_SET_FILE_LIST
#                      is a global
#
# parameters:      train_set - the file listing to be shuffled
#
# returns:         none
#
#-------------------------------------------------------------------------------------

def shuffle_train_set(train_set):
    sorted_file_list = random.shuffle(train_set)
    TRAIN_SET_FILE_LIST = sorted_file_list
    
# Unit test ---------------
#print ('Before Shuffling ->', TRAIN_SET_FILE_LIST)
#shuffle_train_set(TRAIN_SET_FILE_LIST)
#print ('After Shuffling ->', TRAIN_SET_FILE_LIST)

#---------------------------------------------------------------------------------------
# alexnet(width, height, lr): defines the alexnet
#
# parameters:      width - width of the input image
#                  height - height of the input image
#                  lr - learning rate
#
# returns:         none
#
#-------------------------------------------------------------------------------------

def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='features')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', 
                         learning_rate=lr, name='labels')

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH + MODEL_NAME, 
                        tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3, max_checkpoints=1)

    return model

#---------------------------------------------------------------------------------------
# train_conv_net(): runs the train op
#
# parameters:      none
#
# returns:         none
#
#-------------------------------------------------------------------------------------

def train_conv_net():
    
    val_features = []
    val_labels = []
    
    # get train and test batches
    get_train_test_file_list()
    
    # instantiate model
    model = alexnet(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE)
    
    # read in the validation test set
    for j, test_f_in in enumerate(TEST_SET_FILE_LIST):
        if j == 0:
            val_features, val_labels = input_pipeline(test_f_in, PREPROCESSED_DATA_FOLDER)
        else:
            tmp_feature_batch, tmp_label_batch = input_pipeline(test_f_in, 
                                                                PREPROCESSED_DATA_FOLDER)
            val_features = np.concatenate((tmp_feature_batch, val_features), axis=0)
            val_labels = np.concatenate((tmp_label_batch, val_labels), axis=0)

    val_features = np.array(val_features).reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)

    
    
    # start training process
    for i in range(N_TRAIN_STEPS):

        # shuffle the train set files before each step
        shuffle_train_set(TRAIN_SET_FILE_LIST)
        
        # run through every batch in the training set
        for f_in in TRAIN_SET_FILE_LIST:
            
            # read in a batch of features and labels for training
            feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
            feature_batch = feature_batch.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
            #print ('Feature Batch Shape ->', feature_batch.shape)                
                
            # run the fit operation
            model.fit({'features': feature_batch}, {'labels': label_batch}, n_epoch=1, 
                      validation_set=({'features': val_features}, {'labels': val_labels}), 
                      shuffle=True, snapshot_step=None, show_metric=True, 
                      run_id=MODEL_NAME)
    print (val_features)
            
# unit test -----------------------------------
train_conv_net()

