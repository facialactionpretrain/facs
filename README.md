# This is the official repository for the FG2020 paper titled: A Scalable  Approach  for  Facial  Action  Unit  Classifier  Training Using  Noisy  Data  for  Pre-Training


We provide the training code, training dataset, and our pre-trained models

## <b>Dataset</b>
The dataset is organized as follows:
```
/data
  /fine_tune_images
    fold1_test.csv
    fold1_train.csv
    fold2_test.csv
    fold2_train.csv
    fold3_test.csv
    fold3_train.csv
  /meta_data
    ...
  /pre_train_images
    test_label.csv
    train_label.csv
```

The data folders do not contain the actual images, you will need to download them, pre-process the images, and place them into the correponding folders.

#### Pre-Training Dataset
For pre-training we used publicly available MS-Celeb-1M dataset. The dataset contains over 10 million images of 1 million unique individuals retrieved from popular search engines. From this dataset we then randomly sampled over 160K images for annotation to be used for pre-training our model. During the sampling an even gender split was maintained. We then used an off-the-shelf (OpenCV's DNN) face detector to detect and crop out the faces for each of the images. The cropped images were grayscaled and zero padded to maintain a square ratio before using them for pre-training. 

The pre-training dataset can be downloaded at:
https://1drv.ms/u/s!Ar0vPzfI6Urzag_6zS1mYZNpcms?e=61ZCM6

#### Labels
OpenFace 2.0 annotations for each of the pre-training data is provided. The format of the CSV file (test_label.csv and train_label.csv) is as follows: 

```
image,rectsize,gender,AU01,AU02,AU04,AU05,AU06,AU07,AU09,AU10,AU12,AU14,AU15,AU17,AU20,AU23,AU25,AU26,AU28,AU45
```
#### Fine-Tuning
For the fine-tuning stage we used the DISFA dataset (http://mohammadmahoor.com/disfa/). All the frames from each subject video were extracted. We then used an off-the-shelf (OpenCV's DNN) face detector to detect and crop out the faces for each frame. The cropped images were then grayscaled and zero padded to maintain a square ratio before using them to fine-tune our model. 



## <b>Training</b>

### Pre-training
For pre-training the model with the OpenFace 2.0 annotated MS-Celeb-1M subset
```
python src/train.py -d data -r models
```

### Fine-tuning
For fine-tuning with DISFA dataset
```
python src/train.py -d data -r models -ft <pre_trained model location>
```

## <b>Pre-Trained Models</b>

#### The pre-trained models used for fine-tuning:
https://1drv.ms/u/s!Ar0vPzfI6Urzag_6zS1mYZNpcms?e=7e1QmH


#### The final models after fine-tuning:
https://1drv.ms/u/s!Ar0vPzfI6UrzbWQGk_IOwsVibP4?e=c01YgN

# Citation
If you use the dataset or sample code or part of it in your research, please cite the following: