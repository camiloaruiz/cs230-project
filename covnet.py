from __future__ import print_function
from __future__ import division

import numpy as np 
import pandas as pd
import os
import re

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d, conv_1d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import random
from timeit import default_timer as timer

import cv2
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error, roc_auc_score, roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
PREPROCESSED_DATA_FOLDER = 'small/preprocessed/preprocessed'
STAGE1_LABELS = 'tsa_datasets/stage1_labels.csv'
THREAT_ZONE = 1
BATCH_SIZE = 2
EXAMPLES_PER_SUBJECT = 182

FILE_LIST = []
TRAIN_TEST_SPLIT_RATIO = 0.2
TRAIN_SET_FILE_LIST = []
TEST_SET_FILE_LIST = []

IMAGE_DIM = 250
LEARNING_RATE = 1e-4
N_TRAIN_STEPS = 1
TRAIN_PATH_ALEXNET = 'tsa_logs/train/alexnet'
TRAIN_PATH_VGG16 = 'tsa_logs/train/vgg16'
TRAIN_PATH_VGG19 = 'tsa_logs/train/vgg19'
TRAIN_PATH_LOGISTIC = 'tsa_logs/train/logistic'
MODEL_PATH_VGG16 = 'tsa_logs/model/vgg16'
MODEL_PATH_ALEXNET = 'tsa_logs/model/alexnet'
MODEL_PATH_VGG19 = 'tsa_logs/model/vgg19'
MODEL_PATH_LOGISTIC = 'tsa_logs/model/logistic'
MODEL_NAME_ALEXNET = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM, IMAGE_DIM, THREAT_ZONE ))
MODEL_NAME_VGG16 = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('vgg16-v0.1', LEARNING_RATE, IMAGE_DIM, IMAGE_DIM, THREAT_ZONE )) 
MODEL_NAME_VGG19 = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('vgg19-v0.1', LEARNING_RATE, IMAGE_DIM, IMAGE_DIM, THREAT_ZONE ))
MODEL_NAME_LOGISTIC = ('tsa-{}-lr-{}-{}-{}-tz-{}'.format('logistic-v0.1', LEARNING_RATE, IMAGE_DIM, IMAGE_DIM, THREAT_ZONE )) 
# constants
# constants
# constants
COLORMAP = 'pink'
BODY_ZONES = 'tsa_datasets/stage1/body_zones.png'
THREAT_LABELS = 'tsa_datasets/stage1/stage1_labels.csv'


# Divide the available space on an image into 16 sectors. In the [0] image these
# zones correspond to the TSA threat zones.  But on rotated images, the slice
# list uses the sector that best shows the threat zone
sector01_pts = np.array([[0,160],[200,160],[200,230],[0,230]], np.int32)
sector02_pts = np.array([[0,0],[200,0],[200,160],[0,160]], np.int32)
sector03_pts = np.array([[330,160],[512,160],[512,240],[330,240]], np.int32)
sector04_pts = np.array([[350,0],[512,0],[512,160],[350,160]], np.int32)
sector05_pts = np.array([[0,220],[512,220],[512,300],[0,300]], np.int32) # sector 5 is used for both threat zone 5 and 17
sector06_pts = np.array([[0,300],[256,300],[256,360],[0,360]], np.int32)
sector07_pts = np.array([[256,300],[512,300],[512,360],[256,360]], np.int32)
sector08_pts = np.array([[0,370],[225,370],[225,450],[0,450]], np.int32)
sector09_pts = np.array([[225,370],[275,370],[275,450],[225,450]], np.int32)
sector10_pts = np.array([[275,370],[512,370],[512,450],[275,450]], np.int32)
sector11_pts = np.array([[0,450],[256,450],[256,525],[0,525]], np.int32)
sector12_pts = np.array([[256,450],[512,450],[512,525],[256,525]], np.int32)
sector13_pts = np.array([[0,525],[256,525],[256,600],[0,600]], np.int32)
sector14_pts = np.array([[256,525],[512,525],[512,600],[256,600]], np.int32)
sector15_pts = np.array([[0,600],[256,600],[256,660],[0,660]], np.int32)
sector16_pts = np.array([[256,600],[512,600],[512,660],[256,660]], np.int32)

# crop dimensions, upper left x, y, width, height
sector_crop_list = [[ 50,  50, 250, 250], # sector 1
                    [  0,   0, 250, 250], # sector 2
                    [ 50, 250, 250, 250], # sector 3
                    [250,   0, 250, 250], # sector 4
                    [150, 150, 250, 250], # sector 5/17
                    [200, 100, 250, 250], # sector 6
                    [200, 150, 250, 250], # sector 7
                    [250,  50, 250, 250], # sector 8
                    [250, 150, 250, 250], # sector 9
                    [300, 200, 250, 250], # sector 10
                    [400, 100, 250, 250], # sector 11
                    [350, 200, 250, 250], # sector 12
                    [410,   0, 250, 250], # sector 13
                    [410, 200, 250, 250], # sector 14
                    [410,   0, 250, 250], # sector 15
                    [410, 200, 250, 250], # sector 16
                   ]

# Each element in the zone_slice_list contains the sector to use in the call to roi()
zone_slice_list = [ [ # threat zone 1
                      sector01_pts, sector01_pts, sector01_pts, None, None, None, sector03_pts, sector03_pts, 
                      sector03_pts, sector03_pts, sector03_pts, None, None, sector01_pts, sector01_pts, sector01_pts ],       
                    [ # threat zone 2
                      sector02_pts, sector02_pts, sector02_pts, None, None, None, sector04_pts, sector04_pts, 
                      sector04_pts, sector04_pts, sector04_pts, None, None, sector02_pts, sector02_pts, sector02_pts ],
                    [ # threat zone 3
                      sector03_pts, sector03_pts, sector03_pts, sector03_pts, None, None, sector01_pts, sector01_pts,
                      sector01_pts, sector01_pts, sector01_pts, sector01_pts, None, None, sector03_pts, sector03_pts ],
                    [ # threat zone 4
                      sector04_pts, sector04_pts, sector04_pts, sector04_pts, None, None, sector02_pts, sector02_pts, 
                      sector02_pts, sector02_pts, sector02_pts, sector02_pts, None, None, sector04_pts, sector04_pts ],
                    [ # threat zone 5
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts, sector05_pts, sector05_pts, sector05_pts, sector05_pts,
                      None, None, None, None, None, None, None, None ],
                    [ # threat zone 6
                      sector06_pts, None, None, None, None, None, None, None, 
                      sector07_pts, sector07_pts, sector06_pts, sector06_pts, sector06_pts, sector06_pts, sector06_pts, sector06_pts ],
                    [ # threat zone 7
                      sector07_pts, sector07_pts, sector07_pts, sector07_pts, sector07_pts, sector07_pts, sector07_pts, sector07_pts, 
                      None, None, None, None, None, None, None, None ],
                    [ # threat zone 8
                      sector08_pts, sector08_pts, None, None, None, None, None, sector10_pts, 
                      sector10_pts, sector10_pts, sector10_pts, sector10_pts, sector08_pts, sector08_pts, sector08_pts, sector08_pts ],
                    [ # threat zone 9
                      sector09_pts, sector09_pts, sector08_pts, sector08_pts, sector08_pts, None, None, None,
                      sector09_pts, sector09_pts, None, None, None, None, sector10_pts, sector09_pts ],
                    [ # threat zone 10
                      sector10_pts, sector10_pts, sector10_pts, sector10_pts, sector10_pts, sector08_pts, sector10_pts, None, 
                      None, None, None, None, None, None, None, sector10_pts ],
                    [ # threat zone 11
                      sector11_pts, sector11_pts, sector11_pts, sector11_pts, None, None, sector12_pts, sector12_pts,
                      sector12_pts, sector12_pts, sector12_pts, None, sector11_pts, sector11_pts, sector11_pts, sector11_pts ],
                    [ # threat zone 12
                      sector12_pts, sector12_pts, sector12_pts, sector12_pts, sector12_pts, sector11_pts, sector11_pts, sector11_pts, 
                      sector11_pts, sector11_pts, sector11_pts, None, None, sector12_pts, sector12_pts, sector12_pts ],
                    [ # threat zone 13
                      sector13_pts, sector13_pts, sector13_pts, sector13_pts, None, None, sector14_pts, sector14_pts,
                      sector14_pts, sector14_pts, sector14_pts, None, sector13_pts, sector13_pts, sector13_pts, sector13_pts ],
                    [ # sector 14
                      sector14_pts, sector14_pts, sector14_pts, sector14_pts, sector14_pts, None, sector13_pts, sector13_pts, 
                      sector13_pts, sector13_pts, sector13_pts, None, None, None, None, None ],
                    [ # threat zone 15
                      sector15_pts, sector15_pts, sector15_pts, sector15_pts, None, None, sector16_pts, sector16_pts,
                      sector16_pts, sector16_pts, None, sector15_pts, sector15_pts, None, sector15_pts, sector15_pts ],
                    [ # threat zone 16
                      sector16_pts, sector16_pts, sector16_pts, sector16_pts, sector16_pts, sector16_pts, sector15_pts, sector15_pts, 
                      sector15_pts, sector15_pts, sector15_pts, None, None, None, sector16_pts, sector16_pts ],
                    [ # threat zone 17
                      None, None, None, None, None, None, None, None,
                      sector05_pts, sector05_pts, sector05_pts, sector05_pts, sector05_pts, sector05_pts, sector05_pts, sector05_pts ] ]

# Each element in the zone_slice_list contains the sector to use in the call to roi()
zone_crop_list =  [ [ # threat zone 1
                      sector_crop_list[0], sector_crop_list[0], sector_crop_list[0], None, None, None, 
                      sector_crop_list[2], sector_crop_list[2], sector_crop_list[2], sector_crop_list[2], sector_crop_list[2], 
                      None, None, sector_crop_list[0], sector_crop_list[0], sector_crop_list[0] ],       
                    [ # threat zone 2
                      sector_crop_list[1], sector_crop_list[1], sector_crop_list[1], None, None, None, sector_crop_list[3],
                      sector_crop_list[3], sector_crop_list[3], sector_crop_list[3], sector_crop_list[3], None, None,
                      sector_crop_list[1], sector_crop_list[1], sector_crop_list[1] ],
                    [ # threat zone 3
                      sector_crop_list[2], sector_crop_list[2], sector_crop_list[2], sector_crop_list[2], None, None,
                      sector_crop_list[0], sector_crop_list[0], sector_crop_list[0], sector_crop_list[0], sector_crop_list[0],
                      sector_crop_list[0], None, None, sector_crop_list[2], sector_crop_list[2] ],
                    [ # threat zone 4
                      sector_crop_list[3], sector_crop_list[3], sector_crop_list[3], sector_crop_list[3], None, None,
                      sector_crop_list[1], sector_crop_list[1], sector_crop_list[1], sector_crop_list[1], sector_crop_list[1],
                      sector_crop_list[1], None, None, sector_crop_list[3], sector_crop_list[3] ],
                    [ # threat zone 5
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
                      None, None, None, None, None, None, None, None ],
                    [ # threat zone 6
                      sector_crop_list[5], None, None, None, None, None, None, None, 
                      sector_crop_list[6], sector_crop_list[6], sector_crop_list[5], sector_crop_list[5], sector_crop_list[5],
                      sector_crop_list[5], sector_crop_list[5], sector_crop_list[5] ],
                    [ # threat zone 7
                      sector_crop_list[6], sector_crop_list[6], sector_crop_list[6], sector_crop_list[6], sector_crop_list[6],
                      sector_crop_list[6], sector_crop_list[6], sector_crop_list[6], 
                      None, None, None, None, None, None, None, None ],
                    [ # threat zone 8
                      sector_crop_list[7], sector_crop_list[7], None, None, None, None, None, sector_crop_list[9], 
                      sector_crop_list[9], sector_crop_list[9], sector_crop_list[9], sector_crop_list[9], sector_crop_list[7],
                      sector_crop_list[7], sector_crop_list[7], sector_crop_list[7] ],
                    [ # threat zone 9
                      sector_crop_list[8], sector_crop_list[8], sector_crop_list[7], sector_crop_list[7], sector_crop_list[7], None,
                      None, None, sector_crop_list[8], sector_crop_list[8], None, None, None, None, sector_crop_list[9],
                      sector_crop_list[8] ],
                    [ # threat zone 10
                      sector_crop_list[9], sector_crop_list[9], sector_crop_list[9], sector_crop_list[9], sector_crop_list[9],
                      sector_crop_list[7], sector_crop_list[9], None, 
                      None, None, None, None, None, None, None, sector_crop_list[9] ],
                    [ # threat zone 11
                      sector_crop_list[10], sector_crop_list[10], sector_crop_list[10], sector_crop_list[10], None, None,
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
                      None, sector_crop_list[10], sector_crop_list[10], sector_crop_list[10], sector_crop_list[10] ],
                    [ # threat zone 12
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], sector_crop_list[11],
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], 
                      sector_crop_list[11], sector_crop_list[11], sector_crop_list[11], None, None, sector_crop_list[11],
                      sector_crop_list[11], sector_crop_list[11] ],
                    [ # threat zone 13
                      sector_crop_list[12], sector_crop_list[12], sector_crop_list[12], sector_crop_list[12], None, None,
                      sector_crop_list[13], sector_crop_list[13], sector_crop_list[13], sector_crop_list[13], sector_crop_list[13],
                      None, sector_crop_list[12], sector_crop_list[12], sector_crop_list[12], sector_crop_list[12] ],
                    [ # sector 14
                      sector_crop_list[13], sector_crop_list[13], sector_crop_list[13], sector_crop_list[13], sector_crop_list[13],
                      None, sector_crop_list[13], sector_crop_list[13], 
                      sector_crop_list[12], sector_crop_list[12], sector_crop_list[12], None, None, None, None, None ],
                    [ # threat zone 15
                      sector_crop_list[14], sector_crop_list[14], sector_crop_list[14], sector_crop_list[14], None, None,
                      sector_crop_list[15], sector_crop_list[15],
                      sector_crop_list[15], sector_crop_list[15], None, sector_crop_list[14], sector_crop_list[14], None,
                      sector_crop_list[14], sector_crop_list[14] ],
                    [ # threat zone 16
                      sector_crop_list[15], sector_crop_list[15], sector_crop_list[15], sector_crop_list[15], sector_crop_list[15],
                      sector_crop_list[15], sector_crop_list[14], sector_crop_list[14], 
                      sector_crop_list[14], sector_crop_list[14], sector_crop_list[14], None, None, None, sector_crop_list[15],
                      sector_crop_list[15] ],
                    [ # threat zone 17
                      None, None, None, None, None, None, None, None,
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4], sector_crop_list[4], sector_crop_list[4],
                      sector_crop_list[4], sector_crop_list[4], sector_crop_list[4] ] ]


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



#-----------------------------------------------------------------------------------------------------
# get_subject_labels(infile, subject_id):  lists threat probabilities by zone for a given subject
#
# infile:                                      labels csv file
#
# subject_id:                                  the individual you want the threat zone labels for
#
# returns:                                     a df with the list of zones and contraband (0 or 1)
#
#-----------------------------------------------------------------------------------------------------

def get_subject_labels(infile, subject_id):

    # read labels into a dataframe
    df = pd.read_csv(infile)

    # Separate the zone and subject id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    df = df[['Subject', 'Zone', 'Probability']]
    threat_list = df.loc[df['Subject'] == subject_id]
    
    return threat_list

#-----------------------------------------------------------------------------------------------------
# get_subject_zone_label(zone_num, df):        gets a label for a given subject and zone
#
# zone_num:                                    a 0 based threat zone index
#
# df:                                          a df like that returned from get_subject_labels(...)
#
# returns:                                     [0,1] if contraband is present, [1,0] if it isnt
#
#----------------------------------------------------------------------------------------------------

def get_subject_zone_label(zone_num, df):
    
    # Dict to convert a 0 based threat zone index to the text we need to look up the label
    zone_index = {0: 'Zone1', 1: 'Zone2', 2: 'Zone3', 3: 'Zone4', 4: 'Zone5', 5: 'Zone6', 6: 'Zone7', 7: 'Zone8',
                  8: 'Zone9', 9: 'Zone10', 10: 'Zone11', 11: 'Zone12', 12: 'Zone13', 13: 'Zone14', 14: 'Zone15', 15: 'Zone16',
                  16: 'Zone17'
                 }
    # get the text key from the dictionary
    key = zone_index.get(zone_num)
    
    # select the probability value and make the label
    if df.loc[df['Zone'] == key]['Probability'].values[0] == 1:
        # threat present
        return [0,1]
    else:
        #no threat present
        return [1,0]


#----------------------------------------------------------------------------------
# convert_to_grayscale(img):           converts a ATI scan to grayscale
#
# infile:                              an aps file
#
# returns:                             an image
#----------------------------------------------------------------------------------

def convert_to_grayscale(img):
    # scale pixel values to grayscale
    base_range = np.amax(img) - np.amin(img)
    rescaled_range = 255 - 0
    img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)

    return np.uint8(img_rescaled)


#----------------------------------------------------------------------------------
# spread_spectrum(img):                applies a histogram equalization transformation
#
# img:                                 a single scan
#
# returns:                             a transformed scan
#----------------------------------------------------------------------------------

def spread_spectrum(img):
    img = stats.threshold(img, threshmin=12, newval=0)
    
    # see http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img= clahe.apply(img)
    
    return img


#----------------------------------------------------------------------------------------------
# roi(img, vertices):                  uses vertices to mask the image
#
# img:                                 the image to be masked
#
# vertices:                            a set of vertices that define the region of interest
#
# returns:                             a masked image
#----------------------------------------------------------------------------------------------

def roi(img, vertices):
  
    # blank mask
    mask = np.zeros_like(img)

    # fill the mask
    cv2.fillPoly(mask, [vertices], 255)

    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    
    return masked


#----------------------------------------------------------------------------------------------
# crop(img, crop_list):                uses vertices to mask the image
#
# img:                                 the image to be cropped
#
# crop_list:                           a crop_list entry with [x , y, width, height]
#
# returns:                             a cropped image
#----------------------------------------------------------------------------------------------

def crop(img, crop_list):

    x_coord = crop_list[0]
    y_coord = crop_list[1]
    width = crop_list[2]
    height = crop_list[3]
    
    cropped_img = img[x_coord:x_coord+width, y_coord:y_coord+height]
    
    return cropped_img


#-----------------------------------------------------------------------------------------------------------
# normalize(image): Take segmented tsa image and normalize pixel values to be between 0 and 1
#
# parameters:      image - a tsa scan
#
# returns:         a normalized image
#
#-----------------------------------------------------------------------------------------------------------

def normalize(image):
    MIN_BOUND = 0.0
    MAX_BOUND = 255.0
    
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

#-----------------------------------------------------------------------------------------------------------
# zero_center(image): Shift normalized image data and move the range so it is 0 centered at the PIXEL_MEAN
#
# parameters:      image
#
# returns:         a zero centered image
#
#-----------------------------------------------------------------------------------------------------------

def zero_center(image):
  
    PIXEL_MEAN = 0.014327
    
    image = image - PIXEL_MEAN
    return image


#---------------------------------------------------------------------------------------
# preprocess_tsa_data(): preprocesses the tsa datasets
#
# parameters:      none
#
# returns:         none
#---------------------------------------------------------------------------------------

def preprocess_tsa_data():
    
    #OPTION 1: get a list of all subjects for which there are labels
    # df = pd.read_csv(STAGE1_LABELS)
    # df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    # SUBJECT_LIST = df['Subject'].unique()

    #OPTION 2: get a list of all subjects for whom there is data
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
        for tz_num, threat_zone_x_crop_dims in enumerate(zip(zone_slice_list, 
                                                             zone_crop_list)):

            threat_zone = threat_zone_x_crop_dims[0]
            crop_dims = threat_zone_x_crop_dims[1]

            # get label
            label = np.array(get_subject_zone_label(tz_num, 
                             get_subject_labels(STAGE1_LABELS, subject)))

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
                    rescaled_img = convert_to_grayscale(base_img)
                    print('-> shape {}|mean={}'.format(rescaled_img.shape, 
                                                       rescaled_img.mean()))

                    # spread the spectrum to improve contrast
                    print('-> spreading spectrum')
                    high_contrast_img = spread_spectrum(rescaled_img)
                    print('-> shape {}|mean={}'.format(high_contrast_img.shape,
                                                       high_contrast_img.mean()))

                    # get the masked image
                    print('-> masking image')
                    masked_img = roi(high_contrast_img, threat_zone[img_num])
                    print('-> shape {}|mean={}'.format(masked_img.shape, 
                                                       masked_img.mean()))

                    # crop the image
                    print('-> cropping image')
                    cropped_img = crop(masked_img, crop_dims[img_num])
                    print('-> shape {}|mean={}'.format(cropped_img.shape, 
                                                       cropped_img.mean()))

                    # normalize the image
                    print('-> normalizing image')
                    normalized_img = normalize(cropped_img)
                    print('-> shape {}|mean={}'.format(normalized_img.shape, 
                                                       normalized_img.mean()))

                    # zero center the image
                    print('-> zero centering')
                    zero_centered_img = zero_center(normalized_img)
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
        if ((len(threat_zone_examples) % (BATCH_SIZE * EXAMPLES_PER_SUBJECT)) == 0 and batch_num > 8):
            for tz_num, tz in enumerate(zone_slice_list):
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
    if (len(threat_zone_examples) > 0) and batch_num > 8:
        for tz_num, tz in enumerate(zone_slice_list):

            tz_examples_to_save = []

            # write out the batch and reset
            print(' -> writing: ' + PREPROCESSED_DATA_FOLDER 
                    + 'preprocessed_TSA_scans-tz{}-{}-{}-b{}.npy'.format(tz_num+1, 
                      len(threat_zone_examples[0][1][0]),
                      len(threat_zone_examples[0][1][1]), batch_num))

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
                         learning_rate=lr, name='labels', metric='accuracy')

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH_ALEXNET + MODEL_NAME_ALEXNET, 
                        tensorboard_dir=TRAIN_PATH_ALEXNET, tensorboard_verbose=3, max_checkpoints=1)

    return model


def vgg16(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='features')
    network = conv_2d(network, 64, 3, strides=3, activation='relu')
    network = conv_2d(network, 64, 3, strides=3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = conv_2d(network, 128, 3, strides=3, activation='relu')
    network = conv_2d(network, 128, 3, strides=3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 256, 3, strides=3, activation='relu')
    network = conv_2d(network, 256, 3, strides=3, activation='relu')
    network = conv_2d(network, 256, 3, strides=3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', 
                         learning_rate=lr, name='labels')

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH_VGG16 + MODEL_NAME_VGG16, 
                        tensorboard_dir=TRAIN_PATH_VGG16, tensorboard_verbose=3, max_checkpoints=1)

    return model


def vgg19(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='features')
    network = conv_2d(network, 64, 3, strides=3, activation='relu')
    network = conv_2d(network, 64, 3, strides=3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = conv_2d(network, 128, 3, strides=3, activation='relu')
    network = conv_2d(network, 128, 3, strides=3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 256, 3, strides=3, activation='relu')
    network = conv_2d(network, 256, 3, strides=3, activation='relu')
    network = conv_2d(network, 256, 3, strides=3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = conv_2d(network, 3, 512, strides=3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)
    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', 
                         learning_rate=lr, name='labels', metric=['accuracy','loss'])

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH_VGG19 + MODEL_NAME_VGG19, 
                        tensorboard_dir=TRAIN_PATH_VGG19, tensorboard_verbose=3, max_checkpoints=1)

    return model


def logistic_regression_net(width, height, lr, activation, optimizer):
    network = input_data(shape=[None, width, height, 1], name='features')
    network = fully_connected(network, 2, activation=activation)

    network = regression(network, optimizer=optimizer, loss='categorical_crossentropy', 
                         learning_rate=lr, name='labels')

    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH_LOGISTIC + MODEL_NAME_LOGISTIC, 
                        tensorboard_dir=TRAIN_PATH_LOGISTIC, tensorboard_verbose=3, max_checkpoints=1)

    return model


def get_features_and_labels(): 
    val_features = []
    val_labels = []
    
    # get train and test batches
    get_train_test_file_list()
    
    # read in the validation test set
    for j, test_f_in in enumerate(TEST_SET_FILE_LIST):
        if j == 0:
            val_features, val_labels = input_pipeline(test_f_in, PREPROCESSED_DATA_FOLDER)
        else:
            tmp_feature_batch, tmp_label_batch = input_pipeline(test_f_in, 
                                                                PREPROCESSED_DATA_FOLDER)
            val_features = np.concatenate((tmp_feature_batch, val_features), axis=0)
            val_labels = np.concatenate((tmp_label_batch, val_labels), axis=0)

    val_features = val_features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
    return [val_features, val_labels]

def printGraph(train, test, name, index):
    # sns.set_style("darkgrid")
    epochs = [i+1 for i in range(index)]
    print(name)
    print("Train: ")
    print(train)
    print("Test: ")
    print(test)
    print(epochs)
    plt.plot(epochs, test, 'r', label = "Test")
    plt.plot(epochs, train, 'b', label = "Train")
    plt.legend(loc = "upper left")  
    plt.title(name+" vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.savefig("result_graphs/%s_train_vs_test_%i_epochs.png"%(name, 10))
    plt.show()
    plt.clf()


def get_metrics(predictions, labels, y_conv):
    acc, recall, precision, auc_score, msqe = 0, 0, 0, 0, 0
    num_labels = len(labels)
    for i in range(num_labels):
      acc += accuracy_score(predictions[i], labels[i])
      recall += recall_score(predictions[i], labels[i])
      precision += precision_score(predictions[i], labels[i])
      try:
        auc_score += roc_auc_score(predictions[i], labels[i])
      except ValueError:
        pass

    msqe = mean_squared_error(labels, y_conv)
    auc_score  /= num_labels
    acc  /= num_labels
    precision  /= num_labels
    recall /= num_labels
    return acc, recall, precision, auc_score, msqe

#---------------------------------------------------------------------------------------
# train_conv_net(): runs the train op
#
# parameters:      none
#
# returns:         none
#
#-------------------------------------------------------------------------------------
def train_conv_net(model, model_name, num_epoch):
    # start training process
    train_acc, train_loss, train_recall, train_precision, train_auc, train_msqe = [],[],[],[],[],[]
    test_acc, test_loss, test_recall, test_precision, test_auc, test_msqe = [],[],[],[],[],[]
    count = 0
    for i in range(10):
        # shuffle the train set files before each step
        shuffle_train_set(TRAIN_SET_FILE_LIST)
        print(TRAIN_SET_FILE_LIST)
        # run through every batch in the training set
        
        for f_in in TRAIN_SET_FILE_LIST:
            # read in a batch of features and labels for training
            feature_batch, label_batch = input_pipeline(f_in, PREPROCESSED_DATA_FOLDER)
            feature_batch = feature_batch.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1)
            print ('Feature Batch Shape ->', feature_batch.shape)                
            print()
            # run the fit operation
            model.fit(feature_batch, label_batch, n_epoch=1, 
                      validation_set=({'features': val_features}, {'labels': val_labels}), 
                      shuffle=True, snapshot_step=None, show_metric=True, 
                      run_id=model_name)
            predictions = model.predict(feature_batch)
            y_conv = predictions
            loss = model.evaluate(feature_batch, label_batch)[0]
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
            acc = accuracy_score(predictions, label_batch)
            msqe = mean_squared_error(label_batch, y_conv)
            train_acc.append(acc)
            train_loss.append(loss)
            train_auc.append(auc)
            train_msqe.append(msqe)

            predictions = model.predict(val_features)
            y_conv = predictions
            loss = model.evaluate(val_features, val_labels)[0]
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
            acc = accuracy_score(predictions, val_labels)
            msqe = mean_squared_error(val_labels, y_conv)
            test_acc.append(acc)
            test_loss.append(loss)
            test_auc.append(auc)
            test_msqe.append(msqe)
            print("Getting count: ")
            print(count)
            printGraph(train_acc, test_acc, "Accuracy", count + 1)
            printGraph(train_loss, test_loss, "Loss", count + 1)
            printGraph(train_msqe, test_msqe, "Mean Squared Error", count + 1)
            count += 1

            


# preprocess_tsa_data()
val_features, val_labels = get_features_and_labels()

# AlexNet, basic model

g_alexnet = tf.Graph()
with g_alexnet.as_default():
    alexnet_model = alexnet(IMAGE_DIM, IMAGE_DIM, 1e-4)
    train_conv_net(alexnet_model, MODEL_NAME_ALEXNET, 10)

# VGG16, basic model
g_vgg16 = tf.Graph()
with g_vgg16.as_default():
    vgg16_model = vgg16(IMAGE_DIM, IMAGE_DIM, 1e-4)
    train_conv_net(vgg16_model, MODEL_NAME_VGG16, 10)

# VGG19, basic model
g_vgg19 = tf.Graph()
with g_vgg19.as_default():
    vgg19_model = vgg19(IMAGE_DIM, IMAGE_DIM, 1e-4)
    train_conv_net(vgg19_model, MODEL_NAME_VGG19, 10)

# Logistic regression, basic model
g_logistic = tf.Graph()
with g_logistic.as_default():
    logistic_regression_model = logistic_regression_net(IMAGE_DIM, IMAGE_DIM, 1e-4, 'softmax', 'momentum')
    train_conv_net(logistic_regression_model, MODEL_NAME_LOGISTIC, 10)

# AlexNet, increasing learning rate
g_alexnet_lr_large = tf.Graph()
with g_alexnet_lr_large.as_default():
    alexnet_model = alexnet(IMAGE_DIM, IMAGE_DIM, 1e-3)
    train_conv_net(alexnet_model, MODEL_NAME_VGG16, 10)

# AlexNet, decreasing learning rate
g_alexnet_lr_small = tf.Graph()
with g_alexnet_lr_small.as_default():
    alexnet_model = alexnet(IMAGE_DIM, IMAGE_DIM, 1e-5)
    train_conv_net(alexnet_model, MODEL_NAME_VGG16, 10)

# Logistic regression, increasing number of epochs
g_logistic_epoch50 = tf.Graph()
with g_logistic_epoch50.as_default():
    logistic_regression_model = logistic_regression_net(IMAGE_DIM, IMAGE_DIM, 1e-4, 'softmax', 'momentum')
    train_conv_net(logistic_regression_model, MODEL_NAME_LOGISTIC, 50)

# Logistic regression, activation function changed to tanh
g_logistic_tanh = tf.Graph()
with g_logistic_tanh.as_default():
    logistic_regression_model = logistic_regression_net(IMAGE_DIM, IMAGE_DIM, 1e-4, 'tanh', 'momentum')
    train_conv_net(logistic_regression_model, MODEL_NAME_LOGISTIC, 10)