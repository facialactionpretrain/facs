import sys
import os
import csv
import numpy as np
import logging
import random as rnd
from collections import namedtuple

from PIL import Image
from rect_util import Rect
import img_util as imgu
import matplotlib.pyplot as plt

def display_summary(train_data_reader, test_data_reader):
    '''
    Summarize the data in a tabular format.
    '''
    FACS_count = train_data_reader.FACS_count


    #########Uncomment this section for pre-training with MS Celeb dataset########
    FACS_header = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10',
                    'AU12', 'AU14','AU15', 'AU17', 'AU20', 'AU23', 'AU25',
                    'AU26', 'AU28', 'AU45']

    #########Uncomment this section for fine-turning with DISFA dataset###########
    # FACS_header = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10',
    #             'AU10', 'AU12', 'AU14','AU15', 'AU17', 'AU20', 'AU23', 'AU25',
    #             'AU26', 'AU28', 'AU45']

    logging.info("{0}\t{1}\t{2}".format("".ljust(10), "Train", "Test"))
    for index in range(FACS_count):
        logging.info("{0}\t{1}\t{2}".format(FACS_header[index].ljust(10), 
                     train_data_reader.per_FACS_count[index], 
                     test_data_reader.per_FACS_count[index]))

class FACSParameters():
    '''
    FACS reader parameters
    '''
    def __init__(self, target_size, width, height, determinisitc=False, shuffle=True):
        self.target_size   = target_size
        self.width         = width
        self.height        = height
        self.determinisitc = determinisitc
        self.shuffle       = shuffle
                     
class FACSReader(object):

    @classmethod
    def create(cls, base_folder, sub_folders, label_file_name, parameters):
        '''
        Factory function that create an instance of FACSReader and load the data form disk.
        '''
        reader = cls(base_folder, sub_folders, label_file_name, parameters)
        reader.load_folders()
        return reader
        
    def __init__(self, base_folder, sub_folders, label_file_name, parameters):
        '''
        The read iterate through all the sub_folders and aggregate all the images and their corresponding labels.        
        '''
        self.base_folder     = base_folder
        self.sub_folders     = sub_folders
        self.label_file_name = label_file_name
        self.FACS_count   = parameters.target_size
        self.width           = parameters.width
        self.height          = parameters.height
        self.shuffle         = parameters.shuffle

        # data augmentation parameters.determinisitc
        if parameters.determinisitc:
            self.max_shift = 0.0
            self.max_scale = 1.0
            self.max_angle = 0.0
            self.max_skew = 0.0
            self.do_flip = False
        else:
            self.max_shift = 0.08
            self.max_scale = 1.05
            self.max_angle = 20.0
            self.max_skew = 0.05
            self.do_flip = True
        
        self.data              = None
        self.per_FACS_count = None
        self.batch_start       = 0
        self.indices           = 0

        self.A = imgu.compute_norm_mat(self.width, self.height)
        
    def has_more(self):
        '''
        Return True if there is more min-batches.
        '''
        if self.batch_start < len(self.data):
            return True
        return False

    def reset(self):
        '''
        Start from beginning for the new epoch.
        '''
        self.batch_start = 0

    def size(self):
        '''
        Return the number of images read by this reader.
        '''
        return len(self.data)
        
    def next_minibatch(self, batch_size):
        '''
        Return the next mini-batch, we do data augmentation during constructing each mini-batch.
        '''
        data_size = len(self.data)
        batch_end = min(self.batch_start + batch_size, data_size)
        current_batch_size = batch_end - self.batch_start
        if current_batch_size < 0:
            raise Exception('Reach the end of the training data.')
        
        inputs = np.empty(shape=(current_batch_size, 1, self.width, self.height), dtype=np.float32)
        targets = np.empty(shape=(current_batch_size, self.FACS_count), dtype=np.float32)
        for idx in range(self.batch_start, batch_end):
            index = self.indices[idx]
            distorted_image = imgu.distort_img(self.data[index][1], 
                                               self.data[index][3], 
                                               self.width, 
                                               self.height, 
                                               self.max_shift, 
                                               self.max_scale, 
                                               self.max_angle, 
                                               self.max_skew, 
                                               self.do_flip)
            final_image = imgu.preproc_img(distorted_image)

            inputs[idx-self.batch_start]    = final_image
            targets[idx-self.batch_start,:] = self._process_target(self.data[index][2], self.data[index][0])

        self.batch_start += current_batch_size
        return inputs, targets, current_batch_size
        
    def load_folders(self):
        '''
        Load the actual images from disk. While loading, we normalize the input data.
        '''
        self.reset()
        self.data = []
        self.per_FACS_count = np.zeros(self.FACS_count, dtype=np.int)
        
        for folder_name in self.sub_folders: 
            logging.info("Loading %s" % (os.path.join(self.base_folder, folder_name)))
            folder_path = os.path.join(self.base_folder, folder_name)
            in_label_path = os.path.join(folder_path, self.label_file_name)
            with open(in_label_path) as csvfile: 
                FACS_label = csv.reader(csvfile) 
                for row in FACS_label: 
                    # load the image
                    image_path = os.path.join(folder_path, row[0])
                    # Skips any corrupt images
                    try:
                        image_data = Image.open(image_path)
                    except:
                        continue
                    image_data.load()

                    # face rectangle 
                    box = list(map(int, row[1][1:-1].split(',')))
                    face_rc = Rect(box)
                    FACS = list(map(float, row[3:len(row)]))

                    self.data.append((image_path, image_data, FACS, face_rc))
                    # Keeps track for summary printout
                    count_AU = np.array(row[3:len(row)], dtype=np.int32)
                    self.per_FACS_count += count_AU
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _process_target(self, target, name):
        '''
        Multi-label
        '''
        new_target = np.array(target)
        new_target[new_target>0] = 1.0
        return new_target