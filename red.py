import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

carpeta = 'datos_posturas'
archivos = [f for f in os.listdir(carpeta) if f.endswith('.npy')]

X = []
y = []

print("Cargando archivos de posturas...\n")

for archivo in archivos:
    ruta = os.path.join(carpeta, archivo)
    datos = np.load(ruta)
    etiqueta = os.path.splitext(archivo)[0]  

    print(f"  {archivo}: {datos.shape[0]} muestras cargadas.")

    for frame in datos:
        X.append(frame)
        y.append(etiqueta)

X = np.array(X)
y = np.array(y)

print(f"\nTotal de muestras: {X.shape[0]} | Características por muestra: {X.shape[1]}")
print(f"Clases detectadas: {np.unique(y)}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

hist = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrecision en prueba: {acc*100:.2f}%")

model.save('modelo_posturas.h5')
np.save('labels.npy', le.classes_)
print("\nModelo guardado como 'modelo_posturas.h5'")
print("Etiquetas guardadas como 'labels.npy'")
