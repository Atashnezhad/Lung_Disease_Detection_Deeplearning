
# Lung disease detection using Deep learning

### Project Name:
Deep learning for detecting three lung sicknesss using X-Ray iamges

### Description:

This project includes two sub-projects including two class classification and multi-class classification.

Two different data sets gathrered from Kaggle and Github for training two different Convultional Nural Networks (CNN).
The two class classification model can different between normal cases and covid cases. In this case, I had balanced number of images.
In the multiclass classification project, the data was imbalance. I used two common approaches for dealing with imbalance data in image processing including class weight adjustment and over-sampling.


The three lung discesses defination are as follow.

* Pneumocystis pneumonia (PCP) is a serious infection that causes inflammation and fluid buildup in your lungs. It's brought on by a fungus called Pneumocystis jirovecii that spreads through the air. This fungus is very common. Most people's immune systems have fought it off by the time they're 3 or 4 years old.

<p align="left">
  <img src="Assets/Pneu.PNG" >
</p>

* COVID-19 is caused by a coronavirus called SARS-CoV-2. Older adults and people who have severe underlying medical conditions like heart or lung disease or diabetes seem to be at higher risk for developing more serious complications from COVID-19 illness.

<p align="left">
  <img  width="460" height="300" src="Assets/covid.png" >
</p>

* Streptococcus is a genus of gram-positive coccus or spherical bacteria that belongs to the family Streptococcaceae, within the order Lactobacillales, in the phylum Firmicutes. Cell division in streptococci occurs along a single axis, so as they grow, they tend to form pairs or chains that may appear bent or twisted.


**The above definations were gathered from wikipedia and google.**



### Table of Contents:
The project directory tree structure is provided below.
```
├───Assets
├───Codes
│   ├───.ipynb_checkpoints
│   └───CNN_2_classes.ipynb
│   └───CNN_4_classes_Class_Weight_app.ipynb
│   └───Models
├───Dataset
│   ├───Train
│   │   ├───Covid
│   │   └───NORMAL
│   └───Val
│       ├───Covid
│       └───NORMAL
├───Dataset_4_classe
│   ├───Train
│   │   ├───Covid
│   │   ├───NORMAL
│   │   ├───Pneumocystis
│   │   └───Streptococcus
│   └───Val
│       ├───Covid
│       ├───NORMAL
│       ├───Pneumocystis
│       └───Streptococcus
├───Extract and filter images from data set
│   ├───.ipynb_checkpoints
│   └───Dataset
│       ├───Covid
│       ├───NORMAL
│       ├───Pneumocystis
│       └───Streptococcus
└───Figures
```


### Instruction:

**Gathering data:** 

The X-Ray images were gathered from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and [Github](https://github.com/ieee8023/covid-chestxray-datasetrepository).

The data then was divided to two Train and Validation folders.
In this project, we use the deep nutral network to differe the noraml patients from three different sicknesses including Pneumocystis, COVID-19, and Streptococcus.

As it is seen in project directory, the multi class classification data set (Dataset_4_classe) incldued four different sub-folders compare two bi-class classification data set (Data set).






**Assembled Deep Net Model Layers:** 

Any time that you have several images (multiclass classification), use two to three convolution layers. Also,a use softmax as activation for the last layer as I did above (my recommendation but you may test other types either). Note that the categorical_crossentropy isalmost default for multiclass classifiers.

remember that we always use convolution layers for images. the reason is if we use dense layers we will lose positional information in images.

**Prepare Images:**
Using ImageDataGenerator does the normalization (Resacle function does normalization). Then augment the data set for both train and val.
Note that for validation section, I just apply the normalziation part. Next, use flow to apply the data augmention.

* Below dataset images after applying augmentation are seen.

<p align="left">
  <img  width="550" src="Assets/plot_01_assets_2.png" >
</p>



For bi-class classification, the number of images are equal so there is no need for balancing dataset. However, for multi-class classification, I have imbalnace data and I need to consider it to prevent from bias. One way to deal with imbalance data apply class-weight using following stackoverflow three lines code and pass it to the fit function.

```python
counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
```


**Conclusion**

- The CNN developed and was applied for detecting COVID-19 cases from normal. 




* The CNN model different metrics are seen after 10 epoches.
<p align="center">
  <img src="Figures/plot_01_1.png" >
</p>




oversampling
class weigting means multplying loss function
