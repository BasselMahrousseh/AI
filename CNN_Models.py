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

#importing the libraries
#import os 
#import numpy as np
#import glob ,fnmatch, PIL , cv2 
#import matplotlib.pyplot as plt
#from tensorflow.keras.preprocessing import image_dataset_from_directory as dt
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.datasets import mnist
#%matplotlib inline


batch_size = 32
img_height = 20
img_width = 20







def CNN_simple(num_classes):
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
        ])
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    return model

def CNN_simple_no_bias(num_classes):
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(16, 3, padding='same', activation='relu' , use_bias = None ),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu', use_bias = None ),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu', use_bias = None ),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
        ])
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    return model

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
    
    #Train Without LOG
    #model.fit(epochs=EPOCHS , callbacks=[model_checkpoint_callback])

    # The model weights (that are considered the best) are loaded into the model.
    #model.load_weights(checkpoint_filepath)


def CNN_Load_Model(checkpoint_filepath,num_classes):
    #checkpoint_filepath =os.getcwd() + '\Checkpoints\weights.{epoch:02d}-{val_loss:.2f}.h5'
    model = CNN_simple(num_classes)
    model.load_weights(checkpoint_filepath)
    return model

def CNN_Load_Model_no_bias(checkpoint_filepath,num_classes):
    #checkpoint_filepath =os.getcwd() + '\Checkpoints\weights.{epoch:02d}-{val_loss:.2f}.h5'
    model = CNN_simple_no_bias(num_classes)
    model.load_weights(checkpoint_filepath)
    return model


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



def CNN_simple_compile(model):
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])




















def CNN_Mnist_Model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def CNN_Model(X_train, X_test, Y_train, Y_test, nb_filters = 32, batch_size=128, nb_epoch=30, nb_classes=2, do_augment=False, save_file='Models/detector_model.hdf5'):

     # input image dimensions
    img_rows, img_cols = X_train.shape[1], X_train.shape[2]
    
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3) 
    input_shape = (img_rows, img_cols, 1)


    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    # (16, 8, 32)
     
    model.add(Conv2D(nb_filters*2, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters*2, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    # (8, 4, 64) = (2048)
        
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
        
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    if do_augment:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2)
        datagen.fit(X_train)
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=len(X_train), nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test))
    else:
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model.save(save_file)  
