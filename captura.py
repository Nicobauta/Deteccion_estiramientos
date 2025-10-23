import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

if not os.path.exists('datos_posturas'):
    os.makedirs('datos_posturas')

def capturar_postura(nombre_postura, num_frames=200):
    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose()

    datos = []
    print(f"\nCapturando datos para la postura: {nombre_postura}")
    print("Presiona 'q' para finalizar antes de tiempo...\n")

    progress_bar = tqdm(total=num_frames, desc=f"Capturando {nombre_postura}", unit="frames")

    while len(datos) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if results.pose_landmarks:
            puntos = []
            for lm in results.pose_landmarks.landmark:
                puntos.extend([lm.x, lm.y, lm.z])
            datos.append(puntos)
            progress_bar.update(1)

        cv2.imshow(f'Capturando {nombre_postura}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nCaptura detenida manualmente.")
            break

    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()

    datos = np.array(datos)
    ruta_archivo = f"datos_posturas/{nombre_postura}.npy"
    np.save(ruta_archivo, datos)

    print(f"Captura de '{nombre_postura}' finalizada. {len(datos)} fotogramas guardados en: {ruta_archivo}")

while True:
    nombre = input("\nIngresa el nombre de la postura (o escribe 'salir' para terminar): ").strip().lower()
    if nombre == "salir":
        print("\nFinalizando captura de todas las posturas.")
        break

    try:
        num_frames = int(input("Ingresa cuantos fotogramas deseas capturar para esta postura (ej: 200): "))
    except ValueError:
        num_frames = 200
        print("Valor invalido, se usaran 200 frames por defecto.")

    capturar_postura(nombre, num_frames)
