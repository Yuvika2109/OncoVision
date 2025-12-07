import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

print("="*70)
print("🎗️ TRAINING BREAST CANCER CNN MODEL")
print("="*70)

# Check if GPU is available
print(f"\nGPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"TensorFlow Version: {tf.__version__}\n")

# Paths
train_dir = 'data/images/train'
test_dir = 'data/images/test'
model_dir = 'models'

# Create models directory
os.makedirs(model_dir, exist_ok=True)

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8
EPOCHS = 30

print(f"📁 Training data: {train_dir}")
print(f"📁 Test data: {test_dir}")

# Check if data exists
if not os.path.exists(train_dir):
    print(f"❌ ERROR: Training directory not found: {train_dir}")
    exit()

# Data Augmentation
print("\n📊 Setting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\n✅ Training samples: {train_generator.samples}")
print(f"✅ Test samples: {test_generator.samples}")
print(f"✅ Classes found: {train_generator.class_indices}")

# Build CNN Model (SAME as in Image_Classification.py)
print("\n🧠 Building CNN model...")

model = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='conv2d_1'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_2'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_3'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Block 4
    layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_4'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)

print("\n📋 Model Summary:")
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(model_dir, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train
print("\n🚀 Starting training...")
print("="*70)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1
)

# Save final model
final_model_path = os.path.join(model_dir, 'model_cnn.h5')
model.save(final_model_path)
print(f"\n✅ Model saved to: {final_model_path}")

# Evaluate
print("\n📊 Evaluating model on test data...")
test_loss, test_acc, test_auc = model.evaluate(test_generator)
print(f"\n✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Test AUC: {test_auc:.4f}")

# Plot training history
print("\n📈 Generating training plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'training_history.png'), dpi=150)
print(f"✅ Training plots saved to: {model_dir}/training_history.png")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"\n✅ Model ready for deployment!")
print(f"📁 Saved at: {final_model_path}")
print(f"🚀 Run: streamlit run app.py")
