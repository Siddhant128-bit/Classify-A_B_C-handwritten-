import keras
import PIL
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json
import pathlib

data_dir=pathlib.Path('Dataset/')
image_count=len(list(data_dir.glob('*/*.png')))
train_ds=keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(64,64),
        batch_size=32
)

val_ds=keras.preprocessing.image_dataset_from_directory(
       data_dir,
       validation_split=0.2,
       subset='validation',
       seed=123,
       image_size=(64,64),
       batch_size=32
)
class_names=train_ds.class_names
num_class=len(class_names)

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(num_class))
model.compile(optimizer='Adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.summary()

model.fit(train_ds,validation_data=val_ds,epochs=20,validation_steps=20)
model_json=model.to_json()
with open('model.json','w')as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')
