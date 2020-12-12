---
layout: post
title: "Cat or Dog? Image Classification using Convolutional Neural Networks"
description: "Identifying pictures of cats and dogs is a simple task for humans, but a tall order for computers. In this post, we'll look at using a convolutional neural network to classify images."
is_post: true
tags: [python, machine learning]
---

TO DO: DISCUSS VALIDATION LOSS METRIC VS CLASSIFICATION ACCURACY (val loss is diff btw predicted probability and actual class. use example of a cat picture with predicted probability of .49 - classified correctly, but the maximum loss to still be correct)

ALSO: TERMINOLOGY PRIMER OR DELETE

When you look at a picture of a cat or a dog, you don't need to think very hard about which animal is in the picture - you just know. But for a computer, it's a much more difficult task.

I have images of cats and dogs that I got from [this Kaggle competition](https://www.kaggle.com/c/dogs-vs-cats) and I've organized them into a specific directory structure to use with the Keras `flow_from_directory()` method to process the image data. For the training and validation sets, each class has been separated into its own sub-directory. The test set has all the images in one sub-directory since the class is unknown. Note: it's possible to have all images in a single directory and process them using the `flow_from_dataframe()` method, but I found it easier to process them this way.

```
dogs-vs-cats
 |-- test
     |-- test_folder
         |-- 1.jpg
         |-- etc
 |-- train
     |-- cats
         |-- cat.0.jpg
         |-- etc
     |-- dogs
         |-- dog.0.jpg
         |-- etc
 |-- valid
     |-- cats
         |-- cat.1.jpg
         |-- etc
     |-- dogs
         |-- dog.1.jpg
         |-- etc
```

```python
import numpy as np
import pandas as pd
from keras import layers
from keras.layers import Input, Add, Dense, Activation, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model, Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint, EarlyStopping
import scipy.misc
from PIL import Image
import matplotlib.image as mpimg
from matplotlib import transforms
import matplotlib.pyplot as plt

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
import tensorflow as tf
import os
```
### A terminology primer



### Load and pre-process images

First, we need to set up an image generator to iterate through all the images in our directories. The generators for the validation and test sets will simply read in the images and resize them. For the training set though, we will add some noise to the images by randomly rotating, zooming, and flipping them around. This will help the model avoid overfitting, so it will be able to generalize better to a wide range of images and still classify them correctly.

```python
train_datagen = image.ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      zoom_range=0.2,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      horizontal_flip=True,
      fill_mode="nearest"
)
valid_datagen = image.ImageDataGenerator(rescale=1./255)
test_datagen = image.ImageDataGenerator(rescale=1./255)
```
Before we load in any files, we'll set some parameters. We'll start with fixing the image size to 150 x 150 pixels, the batch size to 16, and the number of epochs for training to 50.

```python
IMG_WIDTH, IMG_HEIGHT = 150,150
EPOCHS = 50
BATCH_SIZE = 16
```

Now let's iterate through all the files in our directories and ensure that the generators are working as they should. Here we are also setting the image size and batch size using the values we defined above, and indicating that this is a binary classification problem.

```python
train_generator = train_datagen.flow_from_directory(
    directory=r"./dogs-vs-cats/train/",
    target_size=(IMG_WIDTH,IMG_HEIGHT),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42
)
```

    Found 20000 images belonging to 2 classes.

There are twenty thousand training images, and the generator correctly read them as belonging to two classes since there are two sub-directories within the `train/` folder.

```python
valid_generator = valid_datagen.flow_from_directory(
    directory=r"./dogs-vs-cats/valid/",
    target_size=(IMG_WIDTH,IMG_HEIGHT),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42
)
```

    Found 5000 images belonging to 2 classes.

I've held out five thousand images for the validation set, meaning an 80/20 split between the training and validation data.

```python
test_generator = test_datagen.flow_from_directory(
    directory=r"./dogs-vs-cats/test/",
    target_size=(IMG_WIDTH,IMG_HEIGHT),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)
```

    Found 12500 images belonging to 1 classes.

The Kaggle data included over ten thousand unlabeled images which form the test set. Since we only have one sub-directory, the generator thinks there is only one class. This isn't a problem since we are going to predict the class.

Now, let's get to it!

### Convolutional neural network model with aggressive dropout and early stopping

We will train a small convolutional neural network using dropout and early stopping to try to get the best model. First we'll set up a checkpoint that will save the best model at the end of each epoch, based on the validation loss metric. We'll also set up our early stopping function. The `patience=5` argument means that for each epoch, if the validation loss score does not improve within the next five epochs, the model will stop training and will save the best model up to that point. This is to prevent overfitting on the training data.

```python
mc = ModelCheckpoint(
  'best_model.h5',
  monitor='val_loss',
  mode='min',
  save_best_only=True,
  verbose=1)

es = EarlyStopping(
  monitor='val_loss',
  mode='min',
  verbose=1,
  patience=5)
```
Now we need to set up the structure of our convolutional neural network (CNN) model.
(Information about CNN structure and maybe a diagram here)

```python
model2 = Sequential()
model2.add(Conv2D(32, (3,3), input_shape = (150,150,3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Conv2D(32, (3,3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Conv2D(64, (3,3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Flatten())
model2.add(Dense(64))
model2.add(Activation('relu'))
model2.add(Dropout(0.3))
model2.add(Dense(1))
model2.add(Activation('sigmoid'))

model2.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
```
We'll calculate the step size for the training and validation sets by dividing the total number of images in the data set by the batch size. Once we've set those values, we can call the `fit_generator()` method to actually read in the images and train our model. In the `callbacks` argument, we include the early stopping and model checkpoint functions that we defined earlier .

```python
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model2.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=EPOCHS,
                    callbacks=[es, mc]
)
```
I've truncated the output, but we can see below that the early stopping was applied after epoch 43, meaning the model did not train for the full 50 epochs since it stopped improving.

    ...
    Epoch 00043: val_loss did not improve from 0.25629
    Epoch 00043: early stopping

### Model evaluation

Now that we've trained the model, let's load in the saved model and look at some of the predictions. Since our test set is unlabeled and only Kaggle knows the labels, we can't use that to measure our model's performance. So, we'll evaluate the model on our validation set even though we used it during training. This is not best practice and shouldn't be done in a real setting, since it can inflate the performance metric, making us think our model is better than it actually is and causing it to underperform on new data compared to our expectations.

```python
best_model = load_model('best_model.h5')
```

```python
#evaluate on validation data
eval = best_model.evaluate_generator(generator=valid_generator,
  steps=STEP_SIZE_VALID)
print("Loss:", round(eval[0]),3)
print("Accuracy:", round(eval[1]),3)
```

    Loss: 0.261
    Accuracy: 0.894

When evaluating on the validation set, our accuracy is close to 90%. Not bad for such a simple model! Of course, the real performance is likely to be somewhat worse. The Kaggle competition does not return the accuracy metric when you submit your predictions on the test set, but it does return the loss metric. So we can compare our loss metric above and the loss metric that we get on the test set and see how much they differ.

Below, we'll generate predictions for the test set images and see what our loss metric is from Kaggle.

```python
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()

pred=best_model.predict_generator(
  test_generator,
  steps=STEP_SIZE_TEST,
  verbose=1
)
```
Since the output of the `predict_generator()` method is a probability (in this case, the probability that the image is of class 1: dog), we need to round to get the predicted class if we want to examine our predictions. If the probability is less than 50%, we round down to get 0 (cat), and if it's 50% or more we round up to get 1 (dog).

```python
pred = pred.flatten().tolist()
predicted_class=np.round(pred).astype('int32')
labels = dict({0: 'cat', 1: 'dog'})
predictions = [labels[k] for k in predicted_class]
```

```python
#Kaggle submission
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":pred})
results.to_csv("results.csv",index=False)
```
After submitting to Kaggle, I get a loss metric of .285. That's pretty close to the loss metric we got above after evaluating the model on our validation set, so it seems that our model performance has held up relatively well.

### The fun part: looking at our predictions

Let's look at some pictures and see how well our model did! Remember - humans are naturals at this task so we'll be able to easily tell if our predictions are correct or not just by looking. The code below sets up a function to display 25 pictures and their predicted class (cat or dog), then reads in the first 25 images from our test set using our familiar generator method.

```python
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(label_batch[n])
      plt.axis('off')

image_batch = next(test_datagen.flow_from_directory(
    directory=r"./dogs-vs-cats/test/",
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=25,
    class_mode=None,
    shuffle=False,
    seed=42
))
label_batch = predictions
show_batch(image_batch, label_batch)
```

SHOW IMAGES OF ADORABLE CATS AND DOGS HERE

```python
tucker_generator = test_datagen.flow_from_directory(
    directory=r"./dogs-vs-cats/tucker/",
    target_size=(150,150),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

STEP_SIZE_TUCKER=tucker_generator.n//tucker_generator.batch_size
tucker_generator.reset()
pred_tucker=best_model.predict_generator(tucker_generator,
steps=STEP_SIZE_TUCKER,
verbose=1)

pred_tucker = pred_tucker.flatten().tolist() #from ndarray to single list
predicted_class_tucker=np.round(pred_tucker).astype('int32') #get 0 or 1 for each prediction, in single list
predictions_tucker = [labels[k] for k in predicted_class_tucker]

filenames=tucker_generator.filenames
print(pd.DataFrame({"Photo":filenames,
                      "Predictions":predictions_tucker}))

def show_batch_ten(image_batch, label_batch):
  plt.figure(figsize=(16,6))
  for n in range(10):
      ax = plt.subplot(2,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(label_batch[n])
      plt.axis('off')

image_batch = next(test_datagen.flow_from_directory(
    directory=r"./dogs-vs-cats/tucker/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=10,
    class_mode=None,
    shuffle=False,
    seed=42
))
label_batch = predictions_tucker
show_batch_ten(image_batch, label_batch)
```

### Conclusions
