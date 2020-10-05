import os
import cv2
import random
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, Flatten, Softmax, Activation, Dropout, Conv3D, Dense, MaxPooling2D
import numpy as np


def resize_image(img):
    h = 50
    w = 170
    return cv2.resize(img, (w, h))


irr_folder = "irr"
rel_folder = "rel"
X = []
Y = []
all_paths = []
for im in os.listdir(irr_folder):
    all_paths.append(os.path.join(irr_folder,im))
    # X.append(resize_image(cv2.imread(os.path.join(irr_folder, im))))
    # Y.append(0)

for im in os.listdir(rel_folder):
    all_paths.append(os.path.join(rel_folder,im))
    # X.append(resize_image(cv2.imread(os.path.join(rel_folder, im))))
    # Y.append(1)

random.shuffle(all_paths)
for path in all_paths:
    X.append(resize_image(cv2.imread(path)))
    if path.startswith("irr"):
        Y.append(0)
    else:
        Y.append(1)

X = np.array(X)
Y = np.array(Y)
X = X/255.0
batch_size = 32
print("Init shape",X.shape)
# X = np.reshape(X,(3189,30,150,3))
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(50,170,3),padding='SAME'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),padding='SAME'))
model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),padding='SAME'))
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3),padding='SAME'))
model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
learning_rate = 0.001
model.compile(optimizer=Adam(learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])
mc = ModelCheckpoint('contour_classify_30Sep_2.h5', monitor='val_loss', mode='min', verbose=1)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

model.fit(X, Y,epochs=50,batch_size=batch_size,validation_split=0.1,callbacks=[es,mc])



