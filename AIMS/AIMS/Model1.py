import os
import pretty_midi
from scipy.io import wavfile 
import IPython

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Flatten, Dropout, Activation
from tensorflow.keras import backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import glob

n_x = 79
max_T_x = 1000
sequence_length = 20
T_y_generated = 200



##############################################################
#path = "C:\\Users\\Basel\\source\\repos\\AI\\AIMS\\AIMS"
path = os.getcwd()

#for root, dirs, files in os.walk("."):
       # for file in files:
         # if file.endswith(".mid"):
        #     print(os.path.join(root, file))
##############################################################

# We truncate the duration of each example to the first T_x data

X_list = []
for root, dirs, files in os.walk("."):
  for midiFile in files:
   if midiFile.endswith(".mid"):
    # read the MIDI file
    midi_data = pretty_midi.PrettyMIDI(path + "\\" + midiFile)
    note_l = [note.pitch for note in midi_data.instruments[0].notes]
    # convert to one-hot-encoding
    T_x = len(note_l)
    if T_x > max_T_x:
      T_x = max_T_x
    X_ohe = np.zeros((T_x, n_x))
    for t in range(T_x): 
      X_ohe[t, note_l[t]-1] = 1
    # add to the list  
    X_list.append(X_ohe)
    
print(len(X_list))
print(X_list[0].shape)
print(X_list[1].shape)
print(X_list[2].shape)
plt.figure(figsize=(16, 6))
plt.imshow(X_list[2].T, aspect='auto')
plt.set_cmap('gray_r')
plt.grid(True)
plt.show()

#################################################################






