#%%
import os
import matplotlib.pyplot as plt
from keras import models, layers
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

from keras.preprocessing.image import ImageDataGenerator as idg

datagen = idg(
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

#%%

from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir, fname) for
          fname in os.listdir(train_cats_dir)]

#%%

img_path = fnames[5]

img = image.load_img(img_path, target_size = (150,150))

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0

for batch in datagen.flow(x, batch_size = 1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    plt.show()
    if i % 4 == 0:
        break


#%%
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'RMSprop',
              metrics = ['acc'])

#%%

train_datagen = idg(
    rescale = 1./255,
    rotation_range = 0.4,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = idg(rescale = 1./255)

#%%

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150,150),
    batch_size = 10,
    class_mode = 'binary'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (150,150),
    batch_size = 10,
    class_mode = 'binary'
)

#%%

history = model.fit(train_generator,
                    steps_per_epoch = 100,
                    epochs = 100,
                    validation_data = validation_generator,
                    validation_steps = 50)
#%%
model.save('cats_v_dogs_V2.h5')
#%%

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