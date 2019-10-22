import sys
import time
import os
import math
import csv
import argparse
import numpy as np
import logging
import math
from models import *
from facs import *
import cntk as C

FACS_table =    {
                'AU01'  : 0,
                'AU02'  : 1,
                'AU04'  : 2,
                'AU05'  : 3,
                'AU06'  : 4,
                'AU07'  : 5,
                'AU09'  : 6,
                'AU10'  : 7,
                'AU12'  : 8,
                'AU14'  : 9,
                'AU15'  : 10,
                'AU17'  : 11,
                'AU20'  : 12,
                'AU23'  : 13,
                'AU25'  : 14,
                'AU26'  : 15,
                'AU28'  : 16,
                'AU45'  : 17
                }

#TODO Change train and test folders
# List of folders for training, validation and test.
train_folders = ['pre_train_images']
test_folders  = ['pre_train_images']

# def cost_func(training_mode, prediction, target):
def cost_func(prediction, target):
    '''
    Images can contain multiple FACS AU, each multiple labels weighted exactly the same. We use binary cross entropy loss for the multi-label loss
    '''
    train_loss = None
    train_loss = C.binary_cross_entropy(prediction, target)
    return train_loss
    
def main(base_folder, model_folder, ft_model, model_name='VGG13', max_epochs = 300):

    # create needed folders.
    output_model_path   = os.path.join(model_folder, R'train_results')
    output_model_folder = os.path.join(output_model_path, model_name)
    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    # creating logging file 
    logging.basicConfig(filename = os.path.join(output_model_folder, "train.log"), filemode = 'w', level = logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info("Starting training using {} model and max epochs {}.".format(model_name, max_epochs))

    # create the model
    num_classes = len(FACS_table)
    model       = build_model(num_classes, model_name, ft_model)

    # set the input variables.
    input_var = C.input_variable((1, model.input_height, model.input_width), np.float32)
    label_var = C.input_variable((num_classes), np.float32)
    
    # read FACS dataset.
    logging.info("Loading data...")
    train_params        = FACSParameters(num_classes, model.input_height, model.input_width, False)
    test_params = FACSParameters(num_classes, model.input_height, model.input_width, True)

#TODO Chave data reader to be consistent with the label and test set
    train_data_reader   = FACSReader.create(base_folder, train_folders, "test_label.csv", train_params)
    test_data_reader    = FACSReader.create(base_folder, test_folders, "test_label.csv", test_params)
    
    # print summary of the data.
    display_summary(train_data_reader, test_data_reader)
    
    # get the probalistic output of the model.
    z    = model.model(input_var)
    pred = z
    
    epoch_size     = train_data_reader.size()
    minibatch_size = 32

    # Training config
    lr_per_minibatch       = [model.learning_rate]*20 + [model.learning_rate / 2.0]*20 + [model.learning_rate / 10.0]
    mm_time_constant       = -minibatch_size/np.log(0.9)
    lr_schedule            = C.learning_rate_schedule(lr_per_minibatch, unit=C.UnitType.minibatch, epoch_size=epoch_size)
    mm_schedule            = C.momentum_as_time_constant_schedule(mm_time_constant)

    epoch      = 0
    # loss and error cost
    train_loss = cost_func(pred, label_var)
    pe         = C.binary_cross_entropy(z, label_var)

    # construct the trainer
    learner = C.adam(z.parameters, lr_schedule, mm_schedule)
    trainer = C.Trainer(z, (train_loss, pe), learner)

    # Get minibatches of images to train with and perform model training
    # Make sure to set inital minimum test loss sufficiently high
    min_test_sample_loss    = 1e15

    logging.info("Start training...")
    
    best_epoch = 0
    while epoch < max_epochs: 
        train_data_reader.reset()
        test_data_reader.reset()
        
        # Training 
        start_time = time.time()
        training_loss = 0
        training_sample_loss = 0
        test_sample_loss = 0
        while train_data_reader.has_more():
            images, labels, current_batch_size = train_data_reader.next_minibatch(minibatch_size)

            # Specify the mapping of input variables in the model to actual minibatch data to be trained with
            trainer.train_minibatch({input_var : images, label_var : labels})

            # keep track of statistics.
            training_loss     += trainer.previous_minibatch_loss_average * current_batch_size
            training_sample_loss += trainer.previous_minibatch_evaluation_average * current_batch_size
    
        training_sample_loss /= train_data_reader.size()
     
        while test_data_reader.has_more():
            images, labels, current_batch_size = test_data_reader.next_minibatch(minibatch_size)
            test_sample_loss += trainer.test_minibatch({input_var : images, label_var : labels}) * current_batch_size
            
            test_sample_loss /= test_data_reader.size()
            if test_sample_loss < min_test_sample_loss:
                min_test_sample_loss = test_sample_loss
                trainer.save_checkpoint(os.path.join(output_model_folder, "model_{}".format(best_epoch)))
 
        logging.info("Epoch {}: took {:.3f}s".format(epoch, time.time() - start_time))
        logging.info("  batch training loss:\t{:e}".format(training_loss))
        logging.info("  average training sample loss:\t\t{:.4f}".format(training_sample_loss))
        logging.info("  average test sample loss:\t\t{:.4f}".format(test_sample_loss))
                      
        # create a csv writer to keep track of training progress
        with open(os.path.join(output_model_folder) + '/progress.csv', 'a+', newline='') as csvFile:
            writer=csv.writer(csvFile)
            if not epoch:
                writer.writerow(['epoch', 'batch training_loss', 'avg training sample loss', 'avg test sample loss'])
            writer.writerow([epoch, training_loss, training_sample_loss, test_sample_loss])
        csvFile.close()
        epoch += 1

    logging.info("")
    logging.info("Final test loss:\t\t{:.2f}".format(min_test_sample_loss))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", 
                        "--base_folder", 
                        type = str, 
                        help = "Base folder containing the training and testing data.", 
                        required = True)
    parser.add_argument("-r", 
                        "--model_folder", 
                        type = str, 
                        help = "Model destination folder.", 
                        required = True)
    parser.add_argument("-ft", 
                        "--ft_model", 
                        type = str,
                        default=None,
                        help = "Specify location of model to for fine_tuning")


    args = parser.parse_args()
    # main(args.base_folder, args.model_folder, args.training_mode)
    main(args.base_folder, args.model_folder, args.ft_model)