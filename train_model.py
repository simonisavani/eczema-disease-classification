import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

# === Configuration ===
TRAINING_DIR = "datasets"  # Path to your training dataset (eczema and non_eczema folders)
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = (224, 224)  # MobileNetV2 input size

# === Data Preprocessing ===
# Use ImageDataGenerator to augment and preprocess data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the images to [0, 1]
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'  # Assuming two classes: 'eczema' and 'non_eczema'
)

# === Load Pre-trained Model ===
# Load the MobileNetV2 model pre-trained on ImageNet, without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# === Add Custom Classifier ===
# Add a global average pooling layer and a fully connected layer
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduces the feature maps to a single vector
x = Dense(1024, activation='relu')(x)  # Fully connected layer
predictions = Dense(1, activation='sigmoid')(x)  # Output layer (binary classification)

# === Final Model ===
# Combine base model with the custom classifier
model = Model(inputs=base_model.input, outputs=predictions)

# === Freeze Base Model Layers ===
# Freeze the layers of MobileNetV2 (we will train only the new layers)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# === Train the Model ===
# Use EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# === Save the Model ===
model.save("eczema_model_transfer_learning.h5")
