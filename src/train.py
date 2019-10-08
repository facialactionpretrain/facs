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
train_folders = ['Frame_RightCamera_cropped_120']
valid_folders = ['Frame_RightCamera_cropped_120'] 
test_folders  = ['Frame_RightCamera_cropped_120']

def cost_func(training_mode, prediction, target):
    '''
    We use cross entropy in most mode, except for the multi-label mode, which require treating
    multiple labels exactly the same.
    '''

    #TODO Remove all other training more besided multi_target
    train_loss = None
    if training_mode == 'majority' or training_mode == 'probability' or training_mode == 'crossentropy': 
        # Cross Entropy.
        train_loss = C.negate(C.reduce_sum(C.element_times(target, C.log(prediction)), axis=-1))
    elif training_mode == 'multi_target':
        train_loss = C.binary_cross_entropy(prediction, target)
    return train_loss
    
def main(base_folder, model_folder, training_mode='majority', model_name='VGG13', max_epochs = 300):

    # create needed folders.
    output_model_path   = os.path.join(model_folder, R'unbalanced_allppl_iter1_fold3e')
    output_model_folder = os.path.join(output_model_path, model_name + '_' + training_mode)
    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    # creating logging file 
    logging.basicConfig(filename = os.path.join(output_model_folder, "train.log"), filemode = 'w', level = logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info("Starting with training mode {} using {} model and max epochs {}.".format(training_mode, model_name, max_epochs))

    # create the model
    num_classes = len(FACS_table)
    model       = build_model(num_classes, model_name)

    # set the input variables.
    input_var = C.input_variable((1, model.input_height, model.input_width), np.float32)
    label_var = C.input_variable((num_classes), np.float32)
    
    # read FACS dataset.
    logging.info("Loading data...")
    train_params        = FACSParameters(num_classes, model.input_height, model.input_width, training_mode, False)
    test_and_val_params = FACSParameters(num_classes, model.input_height, model.input_width, "majority", True)

#TODO Chave data reader to be consistent with the label and test set
    train_data_reader   = FACSReader.create(base_folder, train_folders, "../3_foldcv_labels/iter1_fold3_train.csv", train_params)
    val_data_reader     = FACSReader.create(base_folder, valid_folders, "../3_foldcv_labels/iter1_fold3_valid.csv", test_and_val_params)
    test_data_reader    = FACSReader.create(base_folder, test_folders, "../3_foldcv_labels/iter1_fold3_test.csv", test_and_val_params)
    
    # print summary of the data.
    display_summary(train_data_reader, val_data_reader, test_data_reader)
    
    # get the probalistic output of the model.
    z    = model.model(input_var)
    pred = z
    

    epoch_size     = train_data_reader.size()
    minibatch_size = 64

    # Training config
    lr_per_minibatch       = [model.learning_rate]*20 + [model.learning_rate / 2.0]*20 + [model.learning_rate / 10.0]
    mm_time_constant       = -minibatch_size/np.log(0.9)
    lr_schedule            = C.learning_rate_schedule(lr_per_minibatch, unit=C.UnitType.minibatch, epoch_size=epoch_size)
    mm_schedule            = C.momentum_as_time_constant_schedule(mm_time_constant)

    epoch      = 0
    # loss and error cost
    train_loss = cost_func(training_mode, pred, label_var)
    pe         = C.binary_cross_entropy(z, label_var)

    # construct the trainer
    learner = C.adam(z.parameters, lr_schedule, mm_schedule)
    trainer = C.Trainer(z, (train_loss, pe), learner)


    # Get minibatches of images to train with and perform model training
    max_val_accuracy    = 100.0
    final_test_accuracy = 100.0
    best_test_accuracy  = 100.0

    logging.info("Start training...")
    
    best_epoch = 0
    while epoch < max_epochs: 
        train_data_reader.reset()
        val_data_reader.reset()
        test_data_reader.reset()
        
        # Training 
        start_time = time.time()
        training_loss = 0
        training_accuracy = 0
        while train_data_reader.has_more():
            images, labels, current_batch_size = train_data_reader.next_minibatch(minibatch_size)

            # Specify the mapping of input variables in the model to actual minibatch data to be trained with
            trainer.train_minibatch({input_var : images, label_var : labels})


            # keep track of statistics.
            training_loss     += trainer.previous_minibatch_loss_average * current_batch_size
            training_accuracy += trainer.previous_minibatch_evaluation_average * current_batch_size

                
        training_accuracy /= train_data_reader.size()
        training_accuracy = training_accuracy
        
        # Validation
        val_accuracy = 0
        while val_data_reader.has_more():
            images, labels, current_batch_size = val_data_reader.next_minibatch(minibatch_size)
            val_accuracy += trainer.test_minibatch({input_var : images, label_var : labels}) * current_batch_size
            
        val_accuracy /= val_data_reader.size()
        val_accuracy = val_accuracy
        
        # if validation accuracy goes higher, we compute test accuracy
        test_run = False
        if val_accuracy < max_val_accuracy:
            best_epoch = epoch
            max_val_accuracy = val_accuracy

            trainer.save_checkpoint(os.path.join(output_model_folder, "model_{}".format(best_epoch)))

            test_run = True
            test_accuracy = 0
            while test_data_reader.has_more():
                images, labels, current_batch_size = test_data_reader.next_minibatch(minibatch_size)
                test_accuracy += trainer.test_minibatch({input_var : images, label_var : labels}) * current_batch_size
            
            test_accuracy /= test_data_reader.size()
            test_accuracy =  test_accuracy
            final_test_accuracy = test_accuracy
            if final_test_accuracy < best_test_accuracy: 
                best_test_accuracy = final_test_accuracy
 
        logging.info("Epoch {}: took {:.3f}s".format(epoch, time.time() - start_time))
        logging.info("  training loss:\t{:e}".format(training_loss))
        logging.info("  training accuracy:\t\t{:.4f}".format(training_accuracy))
        logging.info("  validation accuracy:\t\t{:.4f}".format(val_accuracy))
            
        if test_run:
            logging.info("  test accuracy:\t\t{:.4f}".format(test_accuracy))
            
        # create a csv writer to keep track of training progress
        with open(os.path.join(output_model_folder) + '/progress.csv', 'a+', newline='') as csvFile:
            writer=csv.writer(csvFile)
            if not epoch:
                writer.writerow(['epoch', 'training_loss', 'training_accuracy', 'val_accuracy', 'test_accuracy'])
            writer.writerow([epoch, training_loss, training_accuracy, val_accuracy, test_accuracy])
        csvFile.close()
        epoch += 1

    logging.info("")
    logging.info("Best validation accuracy:\t\t{:.2f}, epoch {}".format(max_val_accuracy, best_epoch))
    logging.info("Test accuracy corresponding to best validation:\t\t{:.2f}".format(final_test_accuracy))
    logging.info("Best test accuracy:\t\t{:.2f}".format(best_test_accuracy))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", 
                        "--base_folder", 
                        type = str, 
                        help = "Base folder containing the training, validation and testing data.", 
                        required = True)
    parser.add_argument("-r", 
                        "--model_folder", 
                        type = str, 
                        help = "Model folder.", 
                        required = True)
    parser.add_argument("-m", 
                        "--training_mode", 
                        type = str,
                        default='majority',
                        help = "Specify the training mode: majority, probability, crossentropy or multi_target.")

    args = parser.parse_args()
    main(args.base_folder, args.model_folder, args.training_mode)