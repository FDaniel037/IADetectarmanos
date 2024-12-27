import cv2
import mediapipe as mp
import streamlit as st
from PIL import Image
import numpy as np
import time

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables globales
contador_juntas = 0
contador_derecha_baja = 0
contador_pinza_lateral = 0
archivo_txt = "conteo_manos.txt"
posiciones_previas = []
ultima_actualizacion = time.time()
duraciones_ciclos = []
total_movimientos = 0

# Estados y temporizadores
estado_actual = ""
ultimo_estado = ""
tiempo_inicio = 0
tiempo_fin = 0
ciclo_inicio = 0
ciclo_movimientos = []
ciclo_completo = False

def detectar_posicion(mano1, mano2):
    global contador_juntas, contador_derecha_baja, contador_pinza_lateral, ultima_actualizacion, posiciones_previas
    global estado_actual, ultimo_estado, tiempo_inicio, tiempo_fin, ciclo_inicio, ciclo_movimientos, ciclo_completo, duraciones_ciclos, total_movimientos

    estado = ""
    tiempo_actual = time.time()

    if mano1 and mano2:  # Ambas manos detectadas
        # Obtener coordenadas
        x1, y1 = mano1[0], mano1[1]  # Coordenadas mano izquierda
        x2, y2 = mano2[0], mano2[1]  # Coordenadas mano derecha

        # Verificar si la mano derecha está bajada (recoger pescado)
        if x2 > x1 + 50:  # Umbral para movimiento lateral
            contador_derecha_baja += 1
            estado = "Mano derecha bajada"

        # Verificar si las manos están juntas mirando hacia abajo (dejar pescado)
        distancia = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if distancia < 100 and y1 > y2:  # Umbral para considerar las manos juntas mirando hacia abajo
            contador_juntas += 1
            estado = "Manos juntas mirando hacia abajo"

        # Verificar si se está haciendo un movimiento de pinza lateral
        if abs(x2 - x1) > 50 and abs(y2 - y1) < 50:  # Umbral para movimiento lateral
            contador_pinza_lateral += 1
            estado = "Pinza lateral"
            print("Pinza lateral detectada")  # Mensaje de depuración

    # Actualizar posiciones previas para el cálculo de movimientos
    posiciones_previas = [(mano1, mano2)]

    # Calcular el tiempo entre movimientos y registrar ciclos completos
    if estado and (estado != estado_actual or tiempo_actual - ultima_actualizacion > 1):  # Reducir el umbral de tiempo
        tiempo_fin = tiempo_actual
        if estado_actual:
            duracion = tiempo_fin - tiempo_inicio
            ciclo_movimientos.append((estado_actual, duracion))
            guardar_en_txt(f"{estado_actual}: {duracion:.2f} segundos")
        estado_actual = estado
        tiempo_inicio = tiempo_actual
        ultima_actualizacion = tiempo_actual

        # Contar el movimiento
        total_movimientos += 1

        # Verificar si se ha completado un ciclo
        if estado == "Mano derecha bajada":
            ciclo_completo = False
            ciclo_inicio = tiempo_actual
            ciclo_movimientos = []

        if estado == "Manos juntas mirando hacia abajo" and ciclo_completo:
            ciclo_duracion = tiempo_actual - ciclo_inicio
            duraciones_ciclos.append(ciclo_duracion)
            guardar_en_txt(f"Total de ciclo: {ciclo_duracion:.2f} segundos")
            ciclo_completo = False

        if estado == "Manos juntas mirando hacia abajo":
            ciclo_completo = True

    return estado

def guardar_en_txt(evento):
    with open(archivo_txt, "a") as file:
        file.write(f"{evento}\n")

def calcular_promedio_ciclos():
    if duraciones_ciclos:
        promedio = sum(duraciones_ciclos) / len(duraciones_ciclos)
        return promedio
    return 0

def calcular_movimientos_por_minuto(duracion_video):
    if duracion_video > 0:
        movimientos_por_minuto = (total_movimientos / duracion_video) * 60
        return round(movimientos_por_minuto)
    return 0

def mostrar_resultados_en_pantalla(promedio_ciclos, movimientos_por_minuto):
    resultado_texto = f"Promedio de ciclos: {promedio_ciclos:.2f} segundos\nMovimientos por minuto: {movimientos_por_minuto}"
    st.write(resultado_texto)

def preprocess_frame(frame):
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un filtro de desenfoque
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar un umbral adaptativo
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Aumentar el contraste
    alpha = 1.5  # Factor de contraste
    beta = 0     # Valor de brillo
    contrasted = cv2.convertScaleAbs(thresh, alpha=alpha, beta=beta)
    
    # Convertir de nuevo a BGR
    preprocessed_frame = cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)
    
    return preprocessed_frame

def apply_hue_offset(image, hue_offset):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]
    img_hsv[:, :, 0] = cv2.add(h, hue_offset)
    img_hsv = cv2.merge([img_hsv[:, :, 0], s, v])
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

def procesar_video(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    duracion_video = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    
    frame_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            promedio_ciclos = calcular_promedio_ciclos()  # Calcular el promedio de ciclos al final del video
            movimientos_por_minuto = calcular_movimientos_por_minuto(duracion_video)  # Calcular movimientos por minuto al final del video
            mostrar_resultados_en_pantalla(promedio_ciclos, movimientos_por_minuto)
            break

        # Redimensionar el frame para que se ajuste a la pantalla vertical
        frame = cv2.resize(frame, (480, 640))

        # Intentar detectar manos sin el filtro
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(frame_rgb)

        # Si no se detectan manos, aplicar el filtro de Hue offset
        if not resultados.multi_hand_landmarks:
            hue_offset = 95
            frame = apply_hue_offset(frame, hue_offset)
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
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

            # Detectar posiciones
            if len(manos) == 2:
                estado = detectar_posicion(manos[0], manos[1])

        # Mostrar el frame procesado en Streamlit
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_placeholder.image(frame_bgr, channels="BGR")

# Interfaz de usuario con Streamlit
st.title("Reconocedor de Posiciones de Manos")

ruta_video = st.file_uploader("Seleccionar video", type=["mp4", "avi"])
if ruta_video is not None:
    # Guardar el archivo subido en el sistema de archivos
    with open("temp_video.mp4", "wb") as f:
        f.write(ruta_video.getbuffer())
    procesar_video("temp_video.mp4")
