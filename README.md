
<p align="center">
  <img width="1700" src="Assets/Head.png" >
</p>


# Artificial intelligence for lung disease detection using chest CT scan images

Artificial intelligence has the potential to help in covid detection using CT scan images from patient's chests. In this project, we apply two convolutional neural networks for image classification. 
Two data sets were gathered from Kaggle and Github for training Convolutional Nural Networks (CNN).
First, a two-class classification model was trained on balanced data (covid vs normal) to differentiate the healthy cases from covid cases.
Second, a neural network was trained to separate four classes including pneumocystis, covid, streptococcus, and normal. 
Two common approaches in image processing to deal with imbalanced data are class weight adjustment and over-sampling. The oversampling was done along with data augmentation (Applying different transformers for this purpose, flip, rotation, zoom) for a four-class classification project. The models were run on the local machine with a few epochs and later uploaded into the google-colab to benefit from Colab GPU. 



# Instruction

**Gathering data:** 
The X-Ray images were gathered from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and [Github](https://github.com/ieee8023/covid-chestxray-datasetrepository). The data ([All_data](https://github.com/Atashnezhad/Lung_Disease_Detection_Deeplearning/tree/Second/All_data)) then was divided into three Train, Validation, and Test folders ([Dataset_augmented_subfolders](https://github.com/Atashnezhad/Lung_Disease_Detection_Deeplearning/tree/Second/Dataset_augmented_subfolders)) (two class classification project). In four class classificaton project, the data ([All_data_4_classes](https://github.com/Atashnezhad/Lung_Disease_Detection_Deeplearning/tree/Second/All_data_4_classes)) was augmented and oversampled ([Dataset_augmented_4_classes](https://github.com/Atashnezhad/Lung_Disease_Detection_Deeplearning/tree/Second/Dataset_augmented_4_classes)) and then was devided into four subfolders including Normal, Covid, Pneumocystis, Streptococcus ([Data_augmented_4_classes_train_test_val](https://github.com/Atashnezhad/Lung_Disease_Detection_Deeplearning/tree/Second/Data_augmented_4_classes_train_test_val)).

**Assembled Deep Net Model Layers:** 
Any time in multiclass classification two to three convolution layers are suggested. Also, a use softmax as activation for the last layer as I did (my recommendation but you may test other types). Note that the categorical_crossentropy is almost default for multiclass classifiers. Remember that we always use convolution layers for images. the reason is if we use dense layers we will lose positional information in images. In four class classification projects, I found that the relu activation function results in higher accuracy. I used Adam optimizer with a learning rate of 0.001.


The CNN Model architecture used for four-class classification is seen below.
```

Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_11 (Conv2D)           (None, 222, 222, 32)      896       
_________________________________________________________________
batch_normalization_11 (Batc (None, 222, 222, 32)      128       
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 220, 220, 64)      18496     
_________________________________________________________________
batch_normalization_12 (Batc (None, 220, 220, 64)      256       
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 110, 110, 64)      0         
_________________________________________________________________
dropout_11 (Dropout)         (None, 110, 110, 64)      0         
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 108, 108, 64)      36928     
_________________________________________________________________
batch_normalization_13 (Batc (None, 108, 108, 64)      256       
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 54, 54, 64)        0         
_________________________________________________________________
dropout_12 (Dropout)         (None, 54, 54, 64)        0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 186624)            0         
_________________________________________________________________
dense_6 (Dense)              (None, 64)                11944000  
_________________________________________________________________
dropout_13 (Dropout)         (None, 64)                0         
_________________________________________________________________
dense_7 (Dense)              (None, 4)                 260       
=================================================================
Total params: 12,001,220
Trainable params: 12,000,900
Non-trainable params: 320
```








**Prepare Images:**
Using ImageDataGenerator does the normalization. 
Note that in the validation and test section, I just applied the normalization. 
In two-class classification, the number of images is equal so there is no need for balancing the dataset. However, for the four-class classification, I have imbalanced data and I need to consider it to prevent bias. 
In four class classification, I augmented and oversampled for all four classes. The Normal and Covid cases were augmented and over-sampled from 190 to 1000 images. The  Pneumocystis and Streptococcus were augmented and over-sampled from 21 and 12 to 1000 images.


# Suggestion

* Balancing data using a generator is one option for dealing with imbalanced data but it is not always the best.
* The weighted objective function can be used as a second option to deal with unbalanced datasets.
* Generally, using either above options results in losing lots of features which results in low model accuracy.
* The results for the two-class and four-class classification projects were promising.
* Different learning rates should be applied to see it will affect the output.


# Results

Both classification models' accuracy reached 80%.
Deep Convolutional Network Network (CNN) Classification results for four classes are seen below.

<p align="center">
  <img  width="2000" src="Assets/Colab/plot_4C_Normal.png" >
    <img  width="2000" src="Assets/Colab/plot_4C_COVID.png" >
    <img  width="2000" src="Assets/Colab/plot_4C_Pneumocystis.png" >
    <img  width="2000" src="Assets/Colab/plot_4C_Streptococcus.png" >
</p>
















<!--
* The **Dataset** two categories are seen below.
<p align="left">
  <img src="Assets/plot_01_assets_1.png" >
</p>

* The **Dataset_4_classe** four categories are seen below.
<p align="left">
  <img src="Assets/plot_01_assets_1_4classes.png" >
</p>

-->




<!--
<p align="center">
  <img width="600" src="Assets/LearningCurvefourClassClassification.png" >
</p>


-->





<!--
**Visualization**

- A visualization using visualkeras library for 4 class classification network is seen below.
<p align="center">
  <img width="500" src="Figures/CNN_4class.png" >
</p>


-->




<!--

- The second Convolutional Nural Networks layers were Visualized below.

<p align="center">
  <img src="Assets/Conv_layer_1_viz.png" >
</p>

You can see that some filters check the edge of images while as we get far from images filters see the roundness of the image.
-->










<!--
* The CNN model different metrics are seen for biclass classification project below.


<p align="left">
  <img width="700" src="Figures/plot_01_1.png" >
</p>



* The CNN model different loss and accuracy metrics are seen for biclass classification project below.
<p align="left">
  <img width="500" src="Figures/FixedclassRes.png" >
</p>


-->








<!--
* Below dataset images after applying augmentation adn balancing are seen.

<p align="left">
  <img  width="2000" src="Assets/plot_01_assets_2_4classes_balanced.png" >
</p>

-->




<!--

### Table of Contents
The project directory tree structure is provided below.
```
├───All_data
├───All_data_4_classes
├───Assets
├───Codes
├───Dataset_augmented_4_classes
├───Data_augmented_4_classes_train_test_val
│   ├───Test
│   │   ├───COVID
│   │   ├───NORMAL
│   │   ├───Pneumocystis
│   │   └───Streptococcus
│   ├───Train
│   │   ├───COVID
│   │   ├───NORMAL
│   │   ├───Pneumocystis
│   │   └───Streptococcus
│   └───val
│       ├───COVID
│       ├───NORMAL
│       ├───Pneumocystis
│       └───Streptococcus
└───Figures
```

-->






<!--
<p align="center">
  <img width="1000" src="Assets/head_2.png" >
  
</p>

-->



<!--
**CNN model Metrics and Conclusion**

The call back function automatically save the best models taking the best val_acc into account. User can call different saved models and use for analysis.
-->









<!--
One way to deal with imbalanced data applies class-weight using following StackOverflow three lines code and pass it to the fit function.

```python
counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
```
-->




