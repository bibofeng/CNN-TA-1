# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # PHASE 1
#
# ### Traning on APPL
#

# %%
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
import cnn_model
from cnn_model import CNN
import pandas as pd
import matplotlib.pyplot as plt


# %%
temp = CNN()
temp.summary()

temp.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


# %%
print(temp)


# %%

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.75,
                                            min_lr=0.0001)

callbacks = [learning_rate_reduction]


# %%
history = temp.fit_generator(training_dataset,
                             steps_per_epoch=100,
                             epochs=40,
                             validation_data=validation_dataset,
                             validation_steps=50)


# %%

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

fig = plt.figure(figsize=(10, 10), edgecolor='Black')
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(accuracy, label='Training Accuracy', color='Blue')
ax1.plot(val_accuracy, label='Validation Accuracy', color='Red')
ax1.set_title("Training and Validation accuracy", fontsize=20)
ax1.set_ylabel("Accuracy", fontsize=15)
ax1.legend()
ax1.grid('True')

ax2.plot(loss, label='Training Loss', color='Blue')
ax2.plot(val_loss, label='Validation Loss', color='Red')
ax2.set_title("Training and validation loss", fontsize=20)
ax2.set_ylabel("Loss", fontsize=15)
ax2.legend()
ax2.grid('True')

plt.show()


# %%
temp.save('MODEL/type_1/phase_1/')

# %% [markdown]
# # PHASE 2
#
#
# ### Traning on BHARTIARTL, RELIANCE, AMZN
#

# %%

temp = load_model('MODEL/type_1/phase_1/')


# %%

image_generator = ImageDataGenerator(rescale=1/255,
                                     validation_split=0.25)

training_dataset = image_generator.flow_from_directory(directory='Data/BHARTIARTL',
                                                       target_size=(15, 15),
                                                       batch_size=16,
                                                       color_mode='grayscale',
                                                       subset="training",
                                                       class_mode='categorical')


validation_dataset = image_generator.flow_from_directory(directory='Data/BHARTIARTL',
                                                         target_size=(15, 15),
                                                         batch_size=16,
                                                         color_mode='grayscale',
                                                         subset="validation",
                                                         class_mode='categorical')


# %%

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=1,
                                            verbose=1,
                                            factor=0.75,
                                            min_lr=0.0001)

callbacks = [learning_rate_reduction]


# %%
history = temp.fit_generator(training_dataset,
                             steps_per_epoch=100,
                             epochs=40,
                             validation_data=validation_dataset,
                             validation_steps=50)


# %%

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

fig = plt.figure(figsize=(10, 10), edgecolor='Black')
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(accuracy, label='Training Accuracy', color='Blue')
ax1.plot(val_accuracy, label='Validation Accuracy', color='Red')
ax1.set_title("Training and Validation accuracy", fontsize=20)
ax1.set_ylabel("Accuracy", fontsize=15)
ax1.legend()
ax1.set_ylim([0.4, 0.9])
ax1.grid('True')


ax2.plot(loss, label='Training Loss', color='Blue')
ax2.plot(val_loss, label='Validation Loss', color='Red')
ax2.set_title("Training and validation loss", fontsize=20)
ax2.set_ylabel("Loss", fontsize=15)
ax2.legend()
ax2.grid('True')
ax2.set_ylim([0.5, 1.5])

plt.show()


# %%
temp.save('MODEL/type_1/phase_2/')

# %% [markdown]
# # PHASE 3
# ### Traning on NASDAQ & NIFTY500 INDICES

# %%

temp = load_model('MODEL/type_1/phase_2/')


# %%

image_generator = ImageDataGenerator(rescale=1/255,
                                     validation_split=0.25)

training_dataset = image_generator.flow_from_directory(directory='Data/NIFTY500',
                                                       target_size=(15, 15),
                                                       batch_size=16,
                                                       color_mode='grayscale',
                                                       subset="training",
                                                       class_mode='categorical')


validation_dataset = image_generator.flow_from_directory(directory='Data/NIFTY500',
                                                         target_size=(15, 15),
                                                         batch_size=16,
                                                         color_mode='grayscale',
                                                         subset="validation",
                                                         class_mode='categorical')


# %%

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=1,
                                            verbose=1,
                                            factor=0.75,
                                            min_lr=0.0001)

callbacks = [learning_rate_reduction]


# %%
history = temp.fit_generator(training_dataset,
                             steps_per_epoch=100,
                             epochs=40,
                             validation_data=validation_dataset,
                             validation_steps=50)


# %%

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

fig = plt.figure(figsize=(10, 10), edgecolor='Black')
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(accuracy, label='Training Accuracy', color='Blue')
ax1.plot(val_accuracy, label='Validation Accuracy', color='Red')
ax1.set_title("Training and Validation accuracy", fontsize=20)
ax1.set_ylabel("Accuracy", fontsize=15)
ax1.legend()
ax1.set_ylim([0.4, 0.9])
ax1.grid('True')


ax2.plot(loss, label='Training Loss', color='Blue')
ax2.plot(val_loss, label='Validation Loss', color='Red')
ax2.set_title("Training and validation loss", fontsize=20)
ax2.set_ylabel("Loss", fontsize=15)
ax2.legend()
ax2.grid('True')
ax2.set_ylim([0.5, 1.5])

plt.show()


# %%
temp.save('MODEL/type_1/phase_3/')


# %%
training_dataset.class_indices


# %%


# %%
