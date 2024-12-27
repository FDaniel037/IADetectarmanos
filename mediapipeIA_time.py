import cv2
import mediapipe as mp
import streamlit as st
import time
import numpy as np
from PIL import Image

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Variables globales
contador_juntas = 0
contador_derecha_baja = 0
contador_traslados = 0
posiciones_previas = []
ultima_posicion = ""
ultima_actualizacion = time.time()

def detectar_posicion(mano1, mano2):
    global contador_juntas, contador_derecha_baja, contador_traslados, ultima_actualizacion, posiciones_previas, ultima_posicion

    estado = ""
    tiempo_actual = time.time()

    if mano1 and mano2:  # Ambas manos detectadas
        # Obtener coordenadas
        x1, y1 = mano1[0], mano1[1]  # Coordenadas mano izquierda
        x2, y2 = mano2[0], mano2[1]  # Coordenadas mano derecha

        # Verificar si las manos están juntas (distancia pequeña)
        distancia = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if distancia < 100:  # Umbral para considerar las manos juntas
            if ultima_posicion != "Manos Juntos":
                contador_juntas += 1
                ultima_posicion = "Manos Juntos"
                estado = "Manos Juntos"

        # Verificar si las manos se están trasladando juntas
        if posiciones_previas:
            x1_prev, y1_prev = posiciones_previas[0][0]
            x2_prev, y2_prev = posiciones_previas[0][1]
            desplazamiento_izquierda = ((x1 - x1_prev) ** 2 + (y1 - y1_prev) ** 2) ** 0.5
            desplazamiento_derecha = ((x2 - x2_prev) ** 2 + (y2 - y2_prev) ** 2) ** 0.5

            if desplazamiento_izquierda > 20 and desplazamiento_derecha > 20:  # Ajustar el umbral
                if ultima_posicion != "Movimiento de Traslado":
                    contador_traslados += 1
                    ultima_posicion = "Movimiento de Traslado"
                    estado = "Movimiento de Traslado"

        # Verificar si la mano derecha está moviéndose hacia afuera
        if x2 > x1 + 100:  # Umbral para movimiento lateral
            if ultima_posicion != "Mano Derecha Moviendo":
                contador_derecha_baja += 1
                ultima_posicion = "Mano Derecha Moviendo"
                estado = "Mano Derecha Moviendo"

    # Actualizar posiciones previas para el cálculo de movimientos
    posiciones_previas = [(mano1, mano2)]

    return estado

def procesar_video_en_tiempo_real(ruta_video):
    global contador_juntas, contador_derecha_baja, contador_traslados
    cap = cv2.VideoCapture(ruta_video)

    # Muestra video en tiempo real
    stframe = st.empty()  # Creamos un espacio vacío en Streamlit para mostrar el video en tiempo real

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar el frame para que ocupe menos espacio
        frame = cv2.resize(frame, (640, 480))  # Ajustar las dimensiones según sea necesario

        # Convertir frame a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(frame_rgb)
        estado = ""

        # Procesar detecciones
        if resultados.multi_hand_landmarks:
            manos = []
            for hand_landmarks in resultados.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
                y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])
                manos.append((x, y))
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detectar posiciones
            if len(manos) == 2:
                estado = detectar_posicion(manos[0], manos[1])

        # Mostrar contador y estado
        frame = cv2.putText(frame, f"Juntas: {contador_juntas}  Derecha Baja: {contador_derecha_baja}  Traslados: {contador_traslados}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frame = cv2.putText(frame, f"Estado: {estado}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Convertir el frame a formato adecuado para Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        stframe.image(img_pil)

        # Añadir un pequeño retraso para simular la velocidad de fotogramas del video
        time.sleep(0.03)

    cap.release()

def main():
    st.title("Reconocedor de Posiciones de Manos")
    video_file = st.file_uploader("Cargar Video", type=["mp4", "avi"])

    if video_file is not None:
        # Guardar el archivo cargado temporalmente
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.getbuffer())

        # Procesar el video en tiempo real
        procesar_video_en_tiempo_real("temp_video.mp4")

        # Mostrar contadores al final
        st.write(f"Juntas: {contador_juntas}")
        st.write(f"Derecha Baja: {contador_derecha_baja}")
        st.write(f"Traslados: {contador_traslados}")

if __name__ == "__main__":
    main()
