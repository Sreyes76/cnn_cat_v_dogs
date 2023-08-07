#%%

import os, shutil
import tensorflow as tf

#%%
original_ds_dir =  r'C:\Users\Sergio\PycharmProjects\Cat-vs-dog\PetImages'

base_dir = r'C:\Users\Sergio\PycharmProjects\Cat-vs-dog\cat_vs_dog_small'
#os.mkdir(base_dir)

train_dir = base_dir + r'\train'
#os.mkdir(train_dir)

validation_dir = base_dir + r'\validation'
#os.mkdir(val_dir)

test_dir = base_dir + r'\test'
#os.mkdir(test_dir)
#%%
train_cats_dir = os.path.join(train_dir,'train_cats')
train_dogs_dir = os.path.join(train_dir,'train_dogs')
#os.mkdir(train_cats_dir)
#os.mkdir(train_dogs_dir)

val_cats_dir = os.path.join(validation_dir,'val_cats')
val_dogs_dir = os.path.join(validation_dir,'val_dogs')
#os.mkdir(val_cats_dir)
#os.mkdir(val_dogs_dir)

test_cats_dir = os.path.join(test_dir,'test_cats')
test_dogs_dir = os.path.join(test_dir,'test_dogs')
#os.mkdir(test_cats_dir)
#os.mkdir(test_dogs_dir)

#%%
fnames = ['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_ds_dir + r'\Cat',fname)
    dst = os.path.join(train_cats_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_ds_dir + r'\Cat',fname)
    dst = os.path.join(val_cats_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_ds_dir + r'\Cat',fname)
    dst = os.path.join(test_cats_dir,fname)
    shutil.copyfile(src,dst)

fnames = ['{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_ds_dir + r'\Dog', fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_ds_dir + r'\Dog', fname)
    dst = os.path.join(val_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_ds_dir + r'\Dog', fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

#%%
from keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu',
                        input_shape = (150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1,activation = 'sigmoid'))

#%%

from keras import optimizers

model.compile( loss = "binary_crossentropy",
               optimizer = optimizers.RMSprop(learning_rate= 1e-4),
               metrics = 'acc')

#%%
#from keras.preprocessing.image import ImageDataGenerator as idg

#train_datagen = idg(rescale = 1./255)
#test_datagen = idg(rescale = 1./255)

#train_generator = train_datagen.flow_from_directory(
#    train_dir,
#    target_size = (150,150),
#    batch_size = 30,
#    class_mode = 'binary'
#)

#validation_generator =  test_datagen.flow_from_directory(
#    validation_dir,
#    target_size=(150,150),
#    batch_size = 30,
#    class_mode = 'binary'
#)

#%%

import keras.utils
#%%
train_ds = keras.utils.image_dataset_from_directory(
    directory = train_dir,
    labels= 'inferred',
    label_mode= 'binary',
    batch_size = 20,
    image_size=(150,150)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = train_dir,
    labels= 'inferred',
    label_mode= 'binary',
    batch_size = 20,
    image_size=(150,150)
)
#%%
def process(image, label):
    image = tf.cast(image/255.,tf.float32)
    return image, label
#%%
train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)
#%%

for data_batch, labels_batch in train_ds:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

#%%

history = model.fit(
    train_ds,
    steps_per_epoch = 100,
    epochs = 30,
    validation_data = validation_ds,
    validation_steps = 50
)

#%%

model.save('cats_v_dogs_v1.h5')

#%%

import matplotlib.pyplot as plt

acc =  history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs,acc,'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc,'r', label = 'Validation accuracy')
plt.title('Validation v Train accuraccy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo', label = 'Training loss')
plt.plot(epochs, val_loss,'r', label = 'Validation loss')
plt.title('Validation v Train loss')
plt.legend()

plt.show()

#%%
test_ds = keras.utils.image_dataset_from_directory(
    directory = test_dir,
    labels= 'inferred',
    label_mode= 'binary',
    batch_size = 20,
    image_size=(150,150)
)

test_ds = test_ds.map(process)
#%%
model = keras.models.load_model(r'C:\Users\Sergio\PycharmProjects\Cat-vs-dog\cats_v_dogs_v1.h5')
#%%
model.evaluate(test_ds)

