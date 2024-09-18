import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import csv


# modelo    
model_path = "Models/pose_landmarker_heavy.task"

# importaciones de mediapipe    
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose  

# globales
latest_frame = None
processing = False

# csv 
csv_file = open('pose_landmarks.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'landmark_id', 'x', 'y', 'z', 'visibility'])

# dibuja las lineas
def draw_landmarks_manually(image, landmarks, connections):
    height, width, _ = image.shape
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    
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

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_frame, processing
    try:
        frame = output_image.numpy_view()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if result.pose_landmarks:
            # Draw the pose landmarks
            for landmarks in result.pose_landmarks:
                draw_landmarks_manually(frame, landmarks, mp_pose.POSE_CONNECTIONS)

                for idx, landmark in enumerate(landmarks):
                    # Write landmark data to CSV
                    csv_writer.writerow([timestamp_ms, idx, landmark.x, landmark.y, landmark.z, landmark.visibility])

        latest_frame = frame
    except Exception as e:
        print(f"Error in print_result: {e}")
        import traceback
        traceback.print_exc()
    finally:
        processing = False

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

# crea el modelo
with PoseLandmarker.create_from_options(options) as landmarker:
    # captura el video
    cap = cv2.VideoCapture(0)
    
    # seteo de la camara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  
    
    # tiempo de cada frame
    frame_time = 1/30  
    last_time = time.time()

    try:
        while cap.isOpened():
            current_time = time.time()
            delta_time = current_time - last_time

            if delta_time >= frame_time:
                success, frame = cap.read()
                if not success:
                    print("Failed to read frame from webcam")
                    break

                if not processing:
                    processing = True
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    landmarker.detect_async(mp_image, int(current_time * 1000))

                if latest_frame is not None:
                    cv2.imshow('Pose Landmarker', latest_frame)
                else:
                    cv2.imshow('Pose Landmarker', frame)

                last_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # libera la camara
        cap.release()
        # cierra la ventana
        cv2.destroyAllWindows()
        # cierra el csv
        csv_file.close()  

# imprime la version de opencv y mediapipe  
print(f"OpenCV version: {cv2.__version__}")
print(f"MediaPipe version: {mp.__version__}")