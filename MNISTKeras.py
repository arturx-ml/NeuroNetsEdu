import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.layers import Dropout, BatchNormalization, LeakyReLU
from keras import regularizers
import matplotlib.pylab as plt

import pandas as pd

np.random.seed(42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
# Model elements
model = Sequential()

opt = Adam()

##Layers
model.add(Dense(800, input_dim=784,activity_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(200,activity_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dense(10))
model.add(Activation('softmax'))

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.000001, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=128, nb_epoch=250, validation_split=0.2, verbose=2,
                    validation_data=(X_test, y_test), shuffle=True, callbacks=[reduce_lr])
plt.figure()
plt.subplot()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='best')
plt.show()
plt.subplot()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['train','test'],loc='best')
plt.show()
