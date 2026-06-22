import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers
models = tf.keras.models
MobileNetV2 = tf.keras.applications.MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
# --- MISSING IMPORTS ADDED HERE ---
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Model = tf.keras.models.Model
import numpy as np


# --- Configuration ---
data_dir = r"C:\Users\PMLS\Desktop\Main\Final_Heart_Disease_Chatbot\data\balanced_augmented_data"
batch_size = 16 
epochs = 30
img_height, img_width = 224, 224

# 1. Prepare Data Generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False # Keep False for accurate evaluation
)

# 2. Build Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_ecg_model.h5', monitor='val_accuracy', save_best_only=True)

# 4. Train
print("\nStarting Training...")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stop, checkpoint]
)

# 5. --- Evaluation (Precision, Recall, F1-Score) ---
print("\nEvaluating Model...")
# Re-load the best model saved by checkpoint
model.load_weights('best_ecg_model.h5')

Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
cm = confusion_matrix(validation_generator.classes, y_pred)
print(cm)

print('\nClassification Report')
target_names = list(validation_generator.class_indices.keys())
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

# 6. Save Final Model
model.save(r"C:\Users\PMLS\Desktop\Main\Final_Heart_Disease_Chatbot\models\final_heart_disease_model.h5")
print("\nTraining complete! Model saved.")