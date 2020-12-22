---
layout: post
title: "Cat or Dog? Image Classification using Convolutional Neural Networks"
description: "Identifying pictures of cats and dogs is a simple task for humans, but a tall order for computers. In this post, we'll look at using a convolutional neural network to classify images."
is_post: true
tags: [python, machine learning]
---

When you look at a picture of a cat or a dog, you don't need to think very hard about which animal is in the picture &mdash; you just know. But for a computer, it's a much more difficult task.

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
<pre class="out">
Found 20000 images belonging to 2 classes.
</pre>

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
<pre class="out">
Found 5000 images belonging to 2 classes.
</pre>

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
<pre class="out">
Found 12500 images belonging to 1 classes.
</pre>

The Kaggle data included over ten thousand unlabeled images which form the test set. Since we only have one sub-directory, the generator thinks there is only one class. This isn't a problem since we are going to predict the class.

Now, let's get to it!

### Convolutional neural network model with dropout and early stopping

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
Now we need to set up the structure of our convolutional neural network (CNN) model. We start with three rounds of convolutional and max pooling layers, then flatten the outputs into a fully-connected layer with 64 neurons. Then, we have a dropout layer that will "turn off" inputs from 30% of the neurons. Finally, we feed the inputs into a layer with only one neuron (since we are doing binary classification). This will give us our probability of class membership.

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

<pre class="out">
...
Epoch 00043: val_loss did not improve from 0.25629
Epoch 00043: early stopping
</pre>

### Model evaluation

Since our test set is unlabeled and only Kaggle knows the labels, we can't use that to measure our model's performance. So, we'll evaluate the model on our validation set even though we used it during training. This is not best practice and shouldn't be done in a real setting, since it can inflate the performance metric, making us think our model is better than it actually is and causing it to underperform on new data compared to our expectations.

The two metrics we'll look at are **accuracy** and **loss**. Accuracy is easy to understand: what percent of the time did you correctly predict the class? The only thing you need to know to compute the accuracy is the *predicted class* and the *actual class*. Loss is a little bit different: you need the *predicted probability* instead of the predicted class. Loss, also known as binary cross-entropy, is computed using this complicated-looking formula:

$$
H_p(q) = - \frac{1}{N} \sum_{i=1}^{N} y_i\cdot\log(p(y_i))+(1-y_i)\cdot\log(1-p(y_i))
$$

Without getting too deeply into the math behind this metric, essentially it is a measure of how closely the predicted probabilities match the actual class. For example, if an observation belongs to class 1 and the predicted probability is .99, those numbers are very close and the loss will be small. On the other hand, if the probability is .01, the loss will be very large. To see how this differs from accuracy, let's use the following example. Suppose you have two observations of class 1 (dogs in this case). For the first observation, the model's predicted probability is .99 &mdash; it's very sure of its guess. For the second observation, the predicted probability is .51. Both observations are correctly classified as dogs, so it gets a perfect score on the accuracy metric. However, the second observation will have a worse loss score than the first because the model was less "sure" of its prediction.

Now that we've trained the model, let's load in the saved model and look at our evaluation metrics.


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
<pre class="out">
Loss: 0.261
Accuracy: 0.894
</pre>

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
Since the output of the `predict_generator()` method is a probability (in this case, the probability that the image is of class 1, dog), we need to round to get the predicted class if we want to examine our predictions. If the probability is less than 50%, we round down to get 0 (cat), and if it's 50% or more we round up to get 1 (dog).

```python
pred = pred.flatten().tolist()
predicted_class=np.round(pred).astype('int32')
labels = dict({0: 'cat', 1: 'dog'})
predictions = [labels[k] for k in predicted_class]
```

Let's look more closely at the format of the predictions.

```python
filenames=test_generator.filenames
print(pd.DataFrame({"Photo":filenames,
                    "Predictions":predictions,
                    "p(Dog)": np.round(pred,2)}).head())
```
<pre class="out">
                   Photo Predictions       p(Dog)
0      test_folder\1.jpg         dog         0.95
1     test_folder\10.jpg         cat         0.01
2    test_folder\100.jpg         dog         0.60
3   test_folder\1000.jpg         dog         0.99
4  test_folder\10000.jpg         dog         0.70
</pre>    

We can see that for the first two pictures, the model was very sure of the class because the raw predicted probability is very far away from .5. For the middle prediction, the model classified it as a dog but the probability is .6 (meaning the model thought there was a 40% chance of it being a cat) &mdash; not a resounding vote of confidence. I'll save these predictions to a csv in order to submit them to Kaggle and see how the model did.

```python
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":pred})
results.to_csv("results.csv",index=False)
```
After submitting to Kaggle, I get a loss metric of .285. That's pretty close to the loss metric we got above after evaluating the model on our validation set, so it seems that our model performance has held up relatively well.

### The fun part: looking at our predictions

Let's look at some pictures and see how well our model did! Remember &mdash; humans are naturals at this task so we'll be able to easily tell if our predictions are correct or not just by looking. The code below sets up a function to display 25 pictures and their predicted class (cat or dog), then reads in the first 25 images from our test set using our familiar generator method.

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
![](/assets/img/cat-dog-classifier/valid_set_predictions.png)

And now for some *extra* fun, I asked my family to send me pictures of their pets. I put them in a separate directory so I'll follow the same steps I did above and just substitute the new directory path.

```python
family_generator = test_datagen.flow_from_directory(
    directory=r"./dogs-vs-cats/family/",
    target_size=(224,224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)
```
```python
STEP_SIZE_FAMILY=family_generator.n//family_generator.batch_size
family_generator.reset()
pred_family=best_model.predict_generator(family_generator,
steps=STEP_SIZE_FAMILY,
verbose=1)
```
```python
pred_family = pred_family.flatten().tolist() #from ndarray to single list
predicted_class_family=np.round(pred_family).astype('int32') #get 0 or 1 for each prediction, in single list
predictions_family = [labels[k] for k in predicted_class_family]
```

Before we show the images, let's examine our predictions.

```python
filenames=family_generator.filenames
print(pd.DataFrame({"Photo":filenames,
                    "Predictions":predictions_family,
                    "p(Dog)": np.round(pred_family,2)}))
```
<pre class="out">
                     Photo Predictions       p(Dog)
0  test_folder\Bennie1.JPG         dog         0.73
1  test_folder\Cat1JPG.JPG         cat         0.10
2     test_folder\Cat2.JPG         cat         0.19
3     test_folder\Cat5.JPG         cat         0.10
4  test_folder\Harper3.JPG         dog         0.67
5   test_folder\Marty1.JPG         dog         0.93
6   test_folder\Rocky1.JPG         dog         0.65
7    test_folder\Sami1.JPG         cat         0.39
8  test_folder\Tucker1.jpg         dog         0.99
9  test_folder\Tucker2.JPG         cat         0.36
</pre>

I only have ten pictures so I'll redefine the function to display ten predictions.

``` python
def show_batch_ten(image_batch, label_batch):
  plt.figure(figsize=(16,6))
  for n in range(10):
      ax = plt.subplot(2,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(label_batch[n])
      plt.axis('off')

image_batch = next(test_datagen.flow_from_directory(
    directory=r"./dogs-vs-cats/family/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=10,
    class_mode=None,
    shuffle=False,
    seed=42
))

label_batch = predictions_family
show_batch_ten(image_batch, label_batch)
```
![](/assets/img/cat-dog-classifier/fam_pets_predictions.png)

Eight out of ten &mdash; not bad!

### Conclusions

Many real-world image recognition problems use convolutional neural networks (CNNs): facial recognition, medical image diagnostics, even mapping climate change through satellite images. The example in this post is a simplified version of those problems, but the underlying methods are the same.
