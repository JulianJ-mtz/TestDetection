import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from scipy.signal import savgol_filter

# Constantes
MODEL_PATH = "Models/pose_landmarker_full.task"
CSV_FILE_PATH = 'pose_landmarks.csv'
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
FPS = 30
FIST_LANDMARKS = {15: "Izquierda", 16: "Derecha"}  # Puntos de referencia de las muñecas

MAX_DATA_POINTS = 200
PLOT_INTERVAL = 100  # en milisegundos
SMOOTHING_WINDOW = 15
SMOOTHING_ORDER = 3


# Variables globales
latest_frame = None
processing = False
previous_positions = {}
previous_velocities = {}
last_timestamp = 0

acceleration_data = {15: deque(maxlen=MAX_DATA_POINTS), 16: deque(maxlen=MAX_DATA_POINTS)}
time_data = deque(maxlen=MAX_DATA_POINTS)


# Configuración de MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def setup_csv():
    csv_file = open(CSV_FILE_PATH, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'landmark_id', 'x', 'y', 'z', 'visibility', 'acc_x', 'acc_y'])
    return csv_file, csv_writer

# Dibuja las líneas
def draw_landmarks_manually(image, landmarks, connections):
    height, width, _ = image.shape
    
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        
        start_x = int(start_point.x * width)
        start_y = int(start_point.y * height)
        end_x = int(end_point.x * width)
        end_y = int(end_point.y * height)
        
        cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        
        
# Calcula la aceleración
def calculate_acceleration(landmark_id, current_position, timestamp_ms):
    global previous_positions, previous_velocities, last_timestamp
    
    if landmark_id not in previous_positions:
        previous_positions[landmark_id] = current_position[:2]  # Solo X e Y
        previous_velocities[landmark_id] = np.zeros(2)  # Solo X e Y
        return np.zeros(2)  # Retorna aceleración 2D
    
    dt = (timestamp_ms - last_timestamp) / 1000  # Convertir a segundos
    if dt == 0:
        return np.zeros(2)  # Retorna aceleración 2D
    
    current_velocity = (np.array(current_position[:2]) - np.array(previous_positions[landmark_id])) / dt
    acceleration = (current_velocity - previous_velocities[landmark_id]) / dt
    
    previous_positions[landmark_id] = current_position[:2]  # Solo X e Y
    previous_velocities[landmark_id] = current_velocity
    
    return acceleration

def update_acceleration_data(landmark_id, acceleration, timestamp):
    acceleration_data[landmark_id].append((acceleration[0], acceleration[1]))
    if len(time_data) == 0 or timestamp > time_data[-1]:
        time_data.append(timestamp)

def smooth_data(data):
    if len(data) < SMOOTHING_WINDOW:
        return data
    return savgol_filter(data, SMOOTHING_WINDOW, SMOOTHING_ORDER)

def setup_plot():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    lines = {}
    for landmark_id, hand in FIST_LANDMARKS.items():
        lines[landmark_id] = {}
        lines[landmark_id]['x'], = ax1.plot([], [], label=f'Mano {hand} X')
        lines[landmark_id]['y'], = ax2.plot([], [], label=f'Mano {hand} Y')
    
    ax1.set_ylabel('Aceleración X')
    ax2.set_ylabel('Aceleración Y')
    ax2.set_xlabel('Tiempo (ms)')
    ax1.set_ylim(-50, 50)  # Establecer límites del eje y para aceleración X
    ax2.set_ylim(-50, 50)  # Establecer límites del eje y para aceleración Y
    ax1.legend()
    ax2.legend()
    return fig, (ax1, ax2), lines

def update_plot(frame, axes, lines):
    for ax in axes:
        ax.relim()
        ax.autoscale_view(scaley=False)  # Solo autoescalar el eje x
    
    for landmark_id, hand in FIST_LANDMARKS.items():
        data = list(acceleration_data[landmark_id])
        if data:
            x_data, y_data = zip(*data)
            x_smooth = smooth_data(x_data)
            y_smooth = smooth_data(y_data)
            lines[landmark_id]['x'].set_data(time_data, x_smooth)
            lines[landmark_id]['y'].set_data(time_data, y_smooth)
    
    plt.tight_layout()
    return [line for landmark_lines in lines.values() for line in landmark_lines.values()]

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int, csv_writer):
    global latest_frame, processing, last_timestamp
    try:
        frame = output_image.numpy_view()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if result.pose_landmarks:
            for landmarks in result.pose_landmarks:
                draw_landmarks_manually(frame, landmarks, mp_pose.POSE_CONNECTIONS)

                for idx, landmark in enumerate(landmarks):
                    position = [landmark.x, landmark.y]  # Solo X e Y
                    acceleration = [0, 0]  # Inicializa con aceleración 2D
                    
                    if idx in FIST_LANDMARKS:
                        acceleration = calculate_acceleration(idx, position, timestamp_ms)
                        update_acceleration_data(idx, acceleration, timestamp_ms)
                    
                        # Dibuja la aceleración como texto en la imagen
                        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        hand = FIST_LANDMARKS[idx]
                        acc_text = f"Mano {hand} Acc: {acceleration[0]:.2f}, {acceleration[1]:.2f}"
                        cv2.putText(frame, acc_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    csv_writer.writerow([
                        timestamp_ms, idx, landmark.x, landmark.y, landmark.z, 
                        landmark.visibility, acceleration[0], acceleration[1], 0  # Z aceleración siempre 0
                    ])

        latest_frame = frame
        last_timestamp = timestamp_ms

     
    except Exception as e:
        print(f"Error en print_result: {e}")
        import traceback
        traceback.print_exc()
    finally:
        processing = False
        
def main():
    global latest_frame, processing, last_timestamp

    if not os.path.exists(MODEL_PATH):
        print(f"Error: El archivo del modelo no existe en la ruta: {MODEL_PATH}")
        print(f"Directorio actual: {os.getcwd()}")
        print(f"Contenido del directorio 'Models': {os.listdir('Models')}")
        return

    csv_file, csv_writer = setup_csv()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=lambda result, output_image, timestamp_ms: 
            print_result(result, output_image, timestamp_ms, csv_writer),
        num_poses=1
    )

    fig, axes, lines = setup_plot()
    ani = FuncAnimation(fig, update_plot, fargs=(axes, lines), interval=PLOT_INTERVAL, blit=True)

    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)

        frame_time = 1 / FPS
        last_time = time.time()

        try:
            while cap.isOpened():
                current_time = time.time()
                if current_time - last_time >= frame_time:
                    success, frame = cap.read()
                    if not success:
                        print("No se pudo leer el fotograma de la cámara web")
                        break

                    if not processing:
                        processing = True
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                        landmarker.detect_async(mp_image, int(current_time * 1000))

                    cv2.imshow('Pose Landmarker', latest_frame if latest_frame is not None else frame)
                    last_time = current_time

                plt.pause(0.001)  # Permitir tiempo para que se actualice la gráfica

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            csv_file.close()
            plt.ioff()  # Desactivar modo interactivo
            plt.close(fig)

    print(f"Versión de OpenCV: {cv2.__version__}")
    print(f"Versión de MediaPipe: {mp.__version__}")

if __name__ == "__main__":
    main()