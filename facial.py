import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the dataset
data = pd.read_csv('fer2013.csv')

# Preprocess the data
def preprocess_data(data):
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    faces /= 255.0
    emotions = pd.get_dummies(data['emotion']).values
    return faces, emotions

faces, emotions = preprocess_data(data)
X_train, X_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.2, random_state=42)

model = Sequential()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator on the training data
datagen.fit(X_train)

# First convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional layer
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fourth convolutional layer
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the layers
model.add(Flatten())

# Fully connected layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 classes of emotions

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)
# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model checkpoint
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Combine all callbacks
callbacks = [lr_scheduler, early_stopping, model_checkpoint]

# Training the model
batch_size = 64
epochs = 50

history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Function to predict and display emotion
def predict_emotion(face):
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    face = face / 255.0
    emotion_label = np.argmax(model.predict(face))
    return emotion_label

# Example prediction
test_face = X_val[0]
predicted_emotion = predict_emotion(test_face)
emotion_map = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

plt.imshow(test_face.squeeze(), cmap='gray')
plt.title(f'Predicted Emotion: {emotion_map[predicted_emotion]}')
plt.show()
