# this will prevent TF from allocating the whole GPU
from keras import backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Softmax
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Concatenate
from keras import regularizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import model_storage

inputs = Input(shape=(64, 64, 3))

l = Conv2D(32, (3, 3), strides=(2,2),kernel_regularizer=regularizers.l2(0.001))(inputs)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = MaxPooling2D(pool_size=(2, 2))(l)

l = Conv2D(64, (3, 3),strides=(2,2))(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = MaxPooling2D(pool_size=(2, 2))(l)

l = Conv2D(96, (2, 2))(l)
l = BatchNormalization()(l)
l = Activation('relu')(l)
l = MaxPooling2D(pool_size=(2, 2))(l)

flat = Flatten()(l)

l = Dense(128)(flat)
l = Dense(64)(l)
l = Dense(1)(l)
l = Activation(activation='sigmoid', name="final_layer")(l)

model = Model(inputs=inputs, outputs=[l])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
print(model.metrics_names)

generator = ImageDataGenerator(rotation_range = 30, shear_range=0.3, rescale=1./255)
training_gen = generator.flow_from_directory("data/training", target_size=(64, 64), color_mode='rgb', class_mode='binary', batch_size=32, shuffle=True, interpolation='nearest')
validation_gen = generator.flow_from_directory("data/validation", target_size=(64, 64), color_mode='rgb', class_mode='binary', batch_size=32, shuffle=True, interpolation='nearest')

tbCallback = TensorBoard(log_dir='./TB', histogram_freq=0, write_graph=True, write_images=True)
cpCallback = ModelCheckpoint("models\\current_model_best.h5", save_best_only=True, monitor="val_acc", mode="max", save_weights_only=False)
history = model.fit_generator(training_gen, epochs=30, steps_per_epoch=370, validation_data=validation_gen,validation_steps=30, callbacks=[cpCallback, tbCallback])

model_storage.save_model(model, "current_model")
model_storage.save_history(history, "current_model")
model_storage.convert_to_tensorflow("current_model")
model_storage.convert_to_tensorflow("current_model_best")