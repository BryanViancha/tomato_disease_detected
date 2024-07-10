import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from preprocess import load_data

# Cargar los datos
train_data_dir = 'data/train'
val_data_dir = 'data/validation'
X_train, y_train = load_data(train_data_dir)
X_val, y_val = load_data(val_data_dir)

# Construir el modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Guardar el modelo
model.save('models/tomato_disease_model_1.h5')
