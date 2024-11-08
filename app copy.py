


import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
import time
import pandas as pd


# Constants for landmark indices
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
MOUTH_COORDS = [(61, 291), (39, 181), (0, 17), (269, 405)]

# Add shoulder landmark indices for pose detection
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

# Add throat landmark indices (example indices; adjust based on your model)
THROAT_IDX = [0, 17, 18, 19]
# Face Mesh Landmark Indices
THROAT_FACE_IDXS = [0]  # Chin


def distance(point_1, point_2):
    return np.linalg.norm(np.array(point_1) - np.array(point_2))

def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    coords_points = [(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in refer_idxs]
    P2_P6 = distance(coords_points[1], coords_points[5])
    P3_P5 = distance(coords_points[2], coords_points[4])
    P1_P4 = distance(coords_points[0], coords_points[3])
    return (P2_P6 + P3_P5) / (2.0 * P1_P4)

def calculate_avg_ear(landmarks, image_w, image_h):
    left_ear = get_ear(landmarks, LEFT_EYE_IDXS, image_w, image_h)
    right_ear = get_ear(landmarks, RIGHT_EYE_IDXS, image_w, image_h)
    return (left_ear + right_ear) / 2.0

def get_mar(landmarks, refer_idxs, frame_width, frame_height):
    coords_points = [(landmarks[i[0]].x * frame_width, landmarks[i[0]].y * frame_height) for i in refer_idxs]
    vertical_distance = distance(coords_points[0], coords_points[3])
    horizontal_distance = distance(coords_points[1], coords_points[2])
    return vertical_distance / horizontal_distance if horizontal_distance != 0 else 0

def calculate_avg_mar(landmarks, image_w, image_h):
    return get_mar(landmarks, MOUTH_COORDS, image_w, image_h)

def detect_yawn(avg_mar, mar_yawn_threshold, yawn_detected, yawn_start_time, yawn_duration_threshold=1.0, yawn_cooldown=2.0):
    current_time = time.time()
    
    if avg_mar > mar_yawn_threshold:
        if not yawn_detected:
            yawn_start_time = current_time
            yawn_detected = True
        elif current_time - yawn_start_time >= yawn_duration_threshold:
            if current_time - yawn_start_time >= yawn_duration_threshold + yawn_cooldown:
                return 1, False, yawn_start_time  # Count yawn and reset
    else:
        yawn_detected = False
    
    return 0, yawn_detected, yawn_start_time

def plot_landmarks(frame, landmarks, idxs, color=(255, 255, 255)):
    for idx in idxs:
        x = int(landmarks[idx].x * frame.shape[1])
        y = int(landmarks[idx].y * frame.shape[0])
        cv2.circle(frame, (x, y), 3, color, -1)

def plot_mouth_landmarks(frame, landmarks, mouth_coords, color=(255, 255, 255)):
    for idx_pair in mouth_coords:
        x1 = int(landmarks[idx_pair[0]].x * frame.shape[1])
        y1 = int(landmarks[idx_pair[0]].y * frame.shape[0])
        x2 = int(landmarks[idx_pair[1]].x * frame.shape[1])
        y2 = int(landmarks[idx_pair[1]].y * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

def plot_shoulders_landmarks(frame, landmarks, shoulder_idxs, color=(255, 0, 0)):
    for idx in shoulder_idxs:
        x = int(landmarks[idx].x * frame.shape[1])
        y = int(landmarks[idx].y * frame.shape[0])
        cv2.circle(frame, (x, y), 5, color, -1)

def plot_throat_landmarks(frame, landmarks, throat_idxs, color=(0, 255, 0)):
    for idx in throat_idxs:
        x = int(landmarks[idx].x * frame.shape[1])
        y = int(landmarks[idx].y * frame.shape[0])
        cv2.circle(frame, (x, y), 5, color, -1)


def detect_pupil(landmarks, eye_landmarks, frame_width, frame_height):
    eye_coords = [(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in eye_landmarks]
    center_x = int((eye_coords[0][0] + eye_coords[3][0]) / 2)
    center_y = int((eye_coords[1][1] + eye_coords[5][1]) / 2)
    return (center_x, center_y)

def log_data(blink_count, yawn_count, avg_ear, avg_mar):
    header = not pd.io.common.file_exists("user_data_log.csv")
    data = {
        "Blink Count": [blink_count],
        "Yawn Count": [yawn_count],
        "Avg EAR": [avg_ear],
        "Avg MAR": [avg_mar],
        "Timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")]
    }
    df = pd.DataFrame(data)
    try:
        df.to_csv("user_data_log.csv", mode='a', header=header, index=False)
    except Exception as e:
        print(f"Error logging data: {e}")

def check_camera_blocked(results):
    return results.multi_face_landmarks is None

def check_for_glasses(landmarks, frame, area_threshold=100, aspect_ratio_threshold=1.5):
    left_eye_coords = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in LEFT_EYE_IDXS]
    right_eye_coords = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in RIGHT_EYE_IDXS]
    
    left_x, left_y, left_w, left_h = cv2.boundingRect(np.array(left_eye_coords, dtype=np.int32))
    right_x, right_y, right_w, right_h = cv2.boundingRect(np.array(right_eye_coords, dtype=np.int32))
    
    left_area = left_w * left_h
    right_area = right_w * right_h
    
    left_aspect_ratio = float(left_w) / left_h if left_h != 0 else 0
    right_aspect_ratio = float(right_w) / right_h if right_h != 0 else 0

    if (left_area > area_threshold and left_aspect_ratio > aspect_ratio_threshold) or \
       (right_area > area_threshold and right_aspect_ratio > aspect_ratio_threshold):
        return "Wearing Glasses"
    
    return "Not Wearing Glasses"

def calculate_angle(point1, point2, point3):
    """Calculate the angle formed by three points."""
    a = np.linalg.norm(np.array(point2) - np.array(point1))
    b = np.linalg.norm(np.array(point2) - np.array(point3))
    c = np.linalg.norm(np.array(point1) - np.array(point3))
    
    # Use the cosine rule to calculate the angle
    angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
    return np.degrees(angle)


def is_wearing_seatbelt(landmarks, frame_width, frame_height, shoulder_idxs=[11, 12], torso_idxs=[12, 24]):
    shoulder_positions = [(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in shoulder_idxs]
    torso_positions = [(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in torso_idxs]

    shoulder_distance = np.linalg.norm(np.array(shoulder_positions[0]) - np.array(shoulder_positions[1]))
    torso_distance = np.linalg.norm(np.array(torso_positions[0]) - np.array(torso_positions[1]))

    # Calculate angles at the neck
    neck_angle = calculate_angle(torso_positions[0], shoulder_positions[0], shoulder_positions[1])

    # Example conditions for heuristic adjustments
    if shoulder_distance < torso_distance * 0.8 and neck_angle < 150:  # Adjust these values based on testing
        return True
    return False

def is_talking_on_phone(landmarks, hand_landmarks, frame_width, frame_height, proximity_threshold=0.15):
    """
    Detects if the user is talking on a mobile phone by checking if hand landmarks are close to the mouth
    and oriented in a typical phone-holding position.

    Parameters:
    - landmarks: Face landmarks from MediaPipe.
    - hand_landmarks: Detected hand landmarks from MediaPipe.
    - frame_width: Width of the video frame.
    - frame_height: Height of the video frame.
    - proximity_threshold: Fraction of frame height for distance threshold.

    Returns:
    - True if talking on phone is detected, False otherwise.
    """
    mouth_landmarks = [61, 291, 39, 181, 0, 17, 269, 405]
    mouth_coords = [(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in mouth_landmarks]
    mouth_center = np.mean(mouth_coords, axis=0)
    distance_threshold = frame_height * proximity_threshold

    for hand in hand_landmarks:
        for i in range(len(hand.landmark)):
            hand_x = hand.landmark[i].x * frame_width
            hand_y = hand.landmark[i].y * frame_height
            distance = np.linalg.norm(np.array([hand_x, hand_y]) - mouth_center)

            # Check if hand is close to mouth and in typical phone position
            if distance < distance_threshold and hand_y < mouth_center[1]:
                return True  # Detected talking on the phone

    return False  # No phone usage detected


def is_smoking(landmarks, hand_landmarks, frame_width, frame_height, proximity_threshold=0.1):
    """
    Detects if the user is smoking by checking if hand landmarks are close to the mouth.

    Parameters:
    - landmarks: Face landmarks from MediaPipe.
    - hand_landmarks: Detected hand landmarks from MediaPipe.
    - frame_width: Width of the video frame.
    - frame_height: Height of the video frame.
    - proximity_threshold: Fraction of frame height for distance threshold.

    Returns:
    - True if smoking is detected, False otherwise.
    """
    mouth_landmarks = [61, 291, 39, 181, 0, 17, 269, 405]
    mouth_coords = [(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in mouth_landmarks]
    mouth_center = np.mean(mouth_coords, axis=0)

    for hand in hand_landmarks:
        for i in range(len(hand.landmark)):
            hand_x = hand.landmark[i].x * frame_width
            hand_y = hand.landmark[i].y * frame_height
            distance = np.linalg.norm(np.array([hand_x, hand_y]) - mouth_center)

            # Check for proximity to mouth
            if distance < (frame_height * proximity_threshold):
                # Optionally check hand orientation (e.g., if fingers are curled)
                if hand_y < mouth_center[1]:  # Hand is above the mouth
                    return True  # Smoking detected

    return False  # No smoking detected

def main():
    st.set_page_config(page_title="EAR & MAR Detection", layout="wide")

    # Add Pose model for shoulder detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5)

    # Load and apply custom CSS
    with open("styles.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    st.title("Driver Monitoring System")
    st.markdown("""<div class="text">This application uses your webcam to detect the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) in real-time. EAR is used to monitor drowsiness, while MAR can indicate if a person is speaking or yawning.</div>""", unsafe_allow_html=True)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

    # Video Settings Expander
    with st.sidebar.expander("🖥️ Video Settings", expanded=True):
        st.markdown('<div class="expander-header">Video Settings</div>', unsafe_allow_html=True)
        run = st.checkbox('Start Video', value=True)
        show_landmarks = st.checkbox('Display Landmarks', value=True)

    # Detection Options Expander
    with st.sidebar.expander("🔍 Detection Options", expanded=True):
        st.markdown('<div class="expander-header">Detection Options</div>', unsafe_allow_html=True)
        detect_drowsiness = st.checkbox('Enable Drowsiness Detection', value=True)
        check_camera_blocked_option = st.checkbox('Check for Camera Blocked', value=True)
        check_glasses_option = st.checkbox('Check for Glasses', value=True)
        detect_smoking_option = st.checkbox('Detect Smoking', value=True)
        detect_yawning_option = st.checkbox('Enable Yawn Detection', value=True)  # Yawn detection checkbox
        detect_phone_talking_option = st.checkbox('Detect Talking on Phone', value=True)  # New checkbox for phone talking detection
        seatbelt_status = st.checkbox('SeatBelt Status', value=True)

    # Thresholds Expander
    with st.sidebar.expander("⚙️ Thresholds", expanded=True):
        st.markdown('<div class="expander-header">Threshold Settings</div>', unsafe_allow_html=True)
        ear_threshold = st.slider("Drowsiness EAR Threshold", min_value=0.1, max_value=0.5, value=0.2, step=0.01)
        mar_yawn_threshold = st.slider("Yawn MAR Threshold", min_value=0.5, max_value=1.0, value=0.6, step=0.01)
        area_threshold = st.slider("Glasses Area Threshold", min_value=50, max_value=300, value=100, step=1)
        aspect_ratio_threshold = st.slider("Glasses Aspect Ratio Threshold", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

    exit_app = st.sidebar.button("Exit Application", key="exit_button")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open video device.")
        return

    video_placeholder = st.empty()

    blink_count = 0
    previous_ear = 0
    yawn_count = 0
    yawn_detected = False
    last_yawn_time = 0
    last_log_time = time.time()

    while run:
        success, frame = cap.read()
        if not success:
            st.error("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 450))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        hand_results = hands.process(frame_rgb)

        # Process the frame for pose landmarks
        pose_results = pose.process(frame_rgb)

        if check_camera_blocked_option:
            camera_blocked = check_camera_blocked(results)

            if camera_blocked:
                cv2.putText(frame, 'Camera Blocked!', (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                st.warning("Camera is blocked! Please clear the view.")
        else:
            camera_blocked = False

        if not camera_blocked and results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            avg_ear = calculate_avg_ear(landmarks, frame.shape[1], frame.shape[0])
            avg_mar = calculate_avg_mar(landmarks, frame.shape[1], frame.shape[0])

            left_pupil = detect_pupil(landmarks, LEFT_EYE_IDXS, frame.shape[1], frame.shape[0])
            right_pupil = detect_pupil(landmarks, RIGHT_EYE_IDXS, frame.shape[1], frame.shape[0])
            cv2.circle(frame, left_pupil, 5, (0, 255, 255), -1)
            cv2.circle(frame, right_pupil, 5, (0, 255, 255), -1)

            if avg_ear < ear_threshold and previous_ear >= ear_threshold:
                blink_count += 1

            previous_ear = avg_ear

            if detect_yawning_option:  # Check if yawn detection is enabled
                yawn_increment, yawn_detected, last_yawn_time = detect_yawn(
                    avg_mar, mar_yawn_threshold, yawn_detected, last_yawn_time,
                    yawn_duration_threshold=1.0, yawn_cooldown=2.0
                )
                if yawn_increment:
                    yawn_count += 1

            if time.time() - last_log_time >= 5:
                log_data(blink_count, yawn_count, avg_ear, avg_mar)
                last_log_time = time.time()

            cv2.putText(frame, f'Avg EAR: {avg_ear:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Avg MAR: {avg_mar:.2f}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'Blink Count: {blink_count}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f'Yawn Count: {yawn_count}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if check_glasses_option:
                glasses_status = check_for_glasses(landmarks, frame, area_threshold, aspect_ratio_threshold)
                cv2.putText(frame, glasses_status, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            if detect_smoking_option and hand_results.multi_hand_landmarks:
                smoking_detected = is_smoking(landmarks, hand_results.multi_hand_landmarks, frame.shape[1], frame.shape[0])
                if smoking_detected:
                    cv2.putText(frame, 'Smoking Detected!', (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    st.warning("Smoking detected! Please refrain from smoking while driving.")

            if pose_results.pose_landmarks:
                shoulder_landmarks = pose_results.pose_landmarks.landmark
                plot_shoulders_landmarks(frame, shoulder_landmarks, [LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX])
                seatbelt_status = is_wearing_seatbelt(shoulder_landmarks, frame.shape[1], frame.shape[0])
                if seatbelt_status:
                    cv2.putText(frame, 'Seatbelt Fastened!', (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Seatbelt Not Fastened!', (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            if detect_drowsiness and avg_ear < ear_threshold:
                cv2.putText(frame, 'Drowsy Alert!', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                st.warning("Drowsiness detected! Please take a break.")

            # Inside your main while loop after processing landmarks
            if hand_results.multi_hand_landmarks:
                if detect_phone_talking_option:  # Check if talking detection is enabled
                    talking_detected = is_talking_on_phone(landmarks, hand_results.multi_hand_landmarks, frame.shape[1], frame.shape[0])
                    if talking_detected:
                        cv2.putText(frame, 'Talking on Phone!', (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        st.warning("Talking on phone detected! Please focus on driving.")


            if show_landmarks:
                for i in range(len(landmarks)):
                    x = int(landmarks[i].x * frame.shape[1])
                    y = int(landmarks[i].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

                plot_landmarks(frame, landmarks, LEFT_EYE_IDXS)
                plot_landmarks(frame, landmarks, RIGHT_EYE_IDXS)
                plot_mouth_landmarks(frame, landmarks, MOUTH_COORDS)

        video_placeholder.image(frame, channels='BGR', use_column_width=True)

        if exit_app:
            st.success("Exiting the application.")
            break

        time.sleep(0.1)

    cap.release()

if __name__ == "__main__":
    main()






















