import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

print("Cargando modelo y etiquetas...")
model = load_model('modelo_posturas.h5')
labels = np.load('labels.npy', allow_pickle=True)
print(f"Modelo cargado. Clases detectadas: {labels}")

pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
color_texto = (0, 255, 0)
grosor_texto = 2
umbral_confianza = 0.7  

print("\nIniciando deteccion en tiempo real (presiona 'ESC' para salir)...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la camara.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        coords = []
        for landmark in results.pose_landmarks.landmark:
            coords.extend([landmark.x, landmark.y, landmark.z])
        coords = np.array(coords).reshape(1, -1)

        prediction = model.predict(coords, verbose=0)
        prob = np.max(prediction)
        label = labels[np.argmax(prediction)]

        if prob > umbral_confianza:
            texto = f"{label} ({prob*100:.1f}%)"
            color_texto = (0, 255, 0)
        else:
            texto = "Postura no reconocida"
            color_texto = (0, 0, 255)

        cv2.putText(frame, texto, (30, 50), font, 1, color_texto, grosor_texto)

    cv2.imshow('Deteccion de Postura en Tiempo Real', frame)

    if cv2.waitKey(10) & 0xFF == 27:
        print("\nDeteccion finalizada por el usuario.")
        break

cap.release()
cv2.destroyAllWindows()
