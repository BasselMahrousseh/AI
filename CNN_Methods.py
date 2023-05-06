#CNN
import os
#Disable Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} ////3 = INFO, WARNING, and ERROR messages are not printed
#Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import utils,Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator





def CNN_train(model,EPOCHS ,train_ds,val_ds):
    checkpoint_filepath =os.getcwd() + '/Checkpoints/weights.{epoch:02d}-{val_loss:.2f}.h5'
    #checkpoint_filepath = os.getcwd() + r'\\Weights'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        #verbose = 1,
        save_best_only=True)

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback])
    






def CNN_Laod_batch(img_array ,img_path, img_height, img_width):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width),color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)

#######Predict#######
def CNN_single_predict(img_path, img_height, img_width, model,class_names):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width),color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    print(img_array.shape)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

def CNN_batch_predict_Print(batch, model,class_names):
    predictions = model.predict(batch)
    shape = batch.shape.as_list()
    score = []
    for i in range(shape[0]):
        #score[i] = tf.nn.softmax(predictions[i])
        score.append(tf.nn.softmax(predictions[i]))
        print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score[i])], 100 * np.max(score[i])))

def CNN_batch_predict(batch, model,class_names):
    predictions = model.predict(batch)
    shape = batch.shape
    shape = list(shape)
    result = []
    score = []
    for i in range(shape[0]):
        #score[i] = tf.nn.softmax(predictions[i])
        score.append(tf.nn.softmax(predictions[i]))
        result.append(class_names[np.argmax(score[i])])
    result = ''.join(result)
    return result ,score




#######Convert#######
def cam_to_tensor(images):
    images = np.array(images,dtype='int')
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    batch = tf.expand_dims(images, 3) # Create a batch
    return batch


def path_to_tensor(img_path,img_height, img_width):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width),color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    return img_array

def add_to_batch(batch,img_array):
    #batch = img_array.numpy()
    #batch = np.append(batch,img_array,axis=0)
    #batch = tf.convert_to_tensor(batch, dtype=tf.float32)
    #experimental
    batch = tf.experimental.numpy.append(batch, img_array, axis=0)
    return batch

    


#test_image = tf.keras.preprocessing.image.load_img(img_path , color_mode = 'grayscale')
#test_image.show()
#print(type(test_image))
#img_array = tf.keras.preprocessing.image.img_to_array(test_image)
#print(img_array)
#predictions = model.predict(img_array)
#score = tf.nn.softmax(predictions)
