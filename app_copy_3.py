

import numpy as np
import cv2
import mediapipe as mp
import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Any, Optional
from typing import List, Tuple, Dict, Any
from fer import FER  # Import the FER library


# Constants for landmark indices
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
MOUTH_COORDS = [(61, 291), (39, 181), (0, 17), (269, 405)]
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12
THROAT_IDX = [0, 17, 18, 19]
THROAT_FACE_IDXS = [0]


# Initialize the FER detector
emotion_detector = FER()

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

# Function to check seatbelt status
def check_seatbelt(frame):
    height, width, _ = frame.shape

    # Define region of interest (ROI)
    seatbelt_roi = frame[int(height * 0.75):height, int(width * 0.25):int(width * 0.75)]

    # Convert the ROI to HSV color space
    hsv_roi = cv2.cvtColor(seatbelt_roi, cv2.COLOR_BGR2HSV)

    # Define HSV range for the seatbelt color (adjust these values based on the seatbelt color)
    lower_color = np.array([0, 50, 50])   # Example lower range for a dark seatbelt
    upper_color = np.array([179, 255, 255]) # Example upper range for a dark seatbelt

    # Create a mask based on the defined color range
    mask = cv2.inRange(hsv_roi, lower_color, upper_color)

    # Optionally apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Determine if any significant contours are found
    seatbelt_detected = False
    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Adjust this threshold as needed for the size of the seatbelt
                seatbelt_detected = True
                break

    return seatbelt_detected


def recognize_expression(frame):
    """
    Recognize the facial expression from the given frame.

    Args:
        frame (np.ndarray): The captured video frame.

    Returns:
        str: The recognized emotion label.
    """
    # Use FER to detect emotions in the frame
    emotion_analysis = emotion_detector.detect_emotions(frame)

    # Check if any emotions were detected
    if emotion_analysis:
        # Get the dominant emotion
        dominant_emotion = emotion_analysis[0]['emotions']
        # Get the emotion with the highest score
        emotion_label = max(dominant_emotion, key=dominant_emotion.get)
        return emotion_label
    return "No emotion detected"


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

def detect_pupil(landmarks, eye_landmarks, frame_width, frame_height):
    eye_coords = [(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in eye_landmarks]
    center_x = int((eye_coords[0][0] + eye_coords[3][0]) / 2)
    center_y = int((eye_coords[1][1] + eye_coords[5][1]) / 2)
    return (center_x, center_y)


def adjust_brightness_contrast(frame: np.ndarray, brightness: int, contrast: int) -> np.ndarray:
    """
    Adjust the brightness and contrast of a given frame.

    Args:
        frame (np.ndarray): The original video frame.
        brightness (int): The brightness adjustment factor (-100 to 100).
        contrast (int): The contrast adjustment factor (-100 to 100).

    Returns:
        np.ndarray: The adjusted video frame.
    """
    # Clip brightness and contrast values to ensure they are within the valid range
    brightness = np.clip(brightness, -100, 100)
    contrast = np.clip(contrast, -100, 100)

    # Adjust brightness
    frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)

    # Adjust contrast
    if contrast > 0:
        frame = cv2.addWeighted(frame, 1 + contrast / 100, frame, 0, 0)
    else:
        frame = cv2.addWeighted(frame, 1 + contrast / 100, frame, 0, 0)

    return frame

def histogram_equalization(frame: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to enhance the contrast of the frame.

    Args:
        frame (np.ndarray): The original video frame.

    Returns:
        np.ndarray: The enhanced video frame.
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    equalized_frame = cv2.equalizeHist(gray_frame)
    # Convert back to BGR color space
    return cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)


def apply_color_map(frame: np.ndarray) -> np.ndarray:
    """
    Apply a color map to the given frame for better visibility.

    Args:
        frame (np.ndarray): The original video frame.

    Returns:
        np.ndarray: The color-mapped video frame.
    """
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply a color map
    color_mapped_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_JET)
    return color_mapped_frame


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

def check_light_conditions(frame: np.ndarray) -> str:
    """
    Check the light conditions of the captured frame.

    Args:
        frame (np.ndarray): The captured video frame.

    Returns:
        str: Message indicating light conditions (Good, Poor, Very Bright).
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the average brightness
    average_brightness = np.mean(gray_frame)

    # Define brightness thresholds
    poor_light_threshold = 50    # Example value for poor light
    good_light_threshold = 150    # Example value for good light

    # Determine light conditions based on brightness
    if average_brightness < poor_light_threshold:
        return "Poor light conditions. Please increase lighting."
    elif average_brightness < good_light_threshold:
        return "Good light conditions."
    else:
        return "Very bright conditions. Please reduce brightness."


def check_for_glasses(landmarks, frame, area_threshold=100, aspect_ratio_threshold=1.5):
    """
    Check if the subject is wearing glasses based on the bounding boxes of the eyes.

    Args:
        landmarks (list): List of facial landmarks.
        frame (ndarray): The current frame from the webcam.
        area_threshold (float): Minimum area of the eye bounding box to consider.
        aspect_ratio_threshold (float): Minimum aspect ratio of the eye bounding box to consider.

    Returns:
        str: "Wearing Glasses" or "Not Wearing Glasses".
    """
    # Get the coordinates of the left and right eye landmarks
    left_eye_coords = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in LEFT_EYE_IDXS]
    right_eye_coords = [(landmarks[i].x * frame.shape[1], landmarks[i].y * frame.shape[0]) for i in RIGHT_EYE_IDXS]

    # Ensure the eye coordinates are valid
    if not left_eye_coords or not right_eye_coords:
        return "Insufficient Data"

    # Calculate bounding rectangles for both eyes
    left_bounding_rect = cv2.boundingRect(np.array(left_eye_coords, dtype=np.int32))
    right_bounding_rect = cv2.boundingRect(np.array(right_eye_coords, dtype=np.int32))

    # Extract bounding box properties
    left_x, left_y, left_w, left_h = left_bounding_rect
    right_x, right_y, right_w, right_h = right_bounding_rect
    
    # Calculate areas and aspect ratios
    left_area = left_w * left_h
    right_area = right_w * right_h
    
    left_aspect_ratio = (left_w / left_h) if left_h > 0 else 0
    right_aspect_ratio = (right_w / right_h) if right_h > 0 else 0

    # Determine if glasses are being worn based on area and aspect ratio
    wearing_glasses = (
        (left_area > area_threshold and left_aspect_ratio > aspect_ratio_threshold) or 
        (right_area > area_threshold and right_aspect_ratio > aspect_ratio_threshold)
    )
    
    return "Wearing Glasses" if wearing_glasses else "Not Wearing Glasses"


def is_smoking(
    landmarks: List[Any],  # List of face landmarks
    hand_landmarks: Optional[List[Any]],  # List of hand landmarks, optional
    frame_width: int,  # Width of the video frame
    frame_height: int,  # Height of the video frame
    proximity_threshold: float = 0.1,  # Distance threshold as a fraction of the frame height
    mouth_landmarks: List[int] = None  # Indices of mouth landmarks
) -> bool:
    """
    Determine if a person is smoking based on the proximity of hand landmarks to mouth landmarks.

    Args:
        landmarks (List[Any]): List of face landmarks.
        hand_landmarks (Optional[List[Any]]): List of hand landmarks. Can be None.
        frame_width (int): Width of the video frame.
        frame_height (int): Height of the video frame.
        proximity_threshold (float): Distance threshold as a fraction of the frame height.
        mouth_landmarks (List[int]): Indices of the mouth landmarks to use.

    Returns:
        bool: True if smoking is detected, otherwise False.
    """
    
    # Default mouth landmark indices if none provided
    if mouth_landmarks is None:
        mouth_landmarks = [61, 291, 39, 181, 0, 17, 269, 405]

    # Check if landmarks are available
    if len(landmarks) < max(mouth_landmarks):
        return False  # Not enough landmarks to define mouth
    
    # Get mouth coordinates and calculate the mouth center
    mouth_coords = np.array([(landmarks[i].x * frame_width, landmarks[i].y * frame_height) for i in mouth_landmarks])
    mouth_center = np.mean(mouth_coords, axis=0)

    # Return False if no hand landmarks are detected
    if hand_landmarks is None or len(hand_landmarks) == 0:
        return False  # No hands detected

    # Convert hand landmarks to a NumPy array for vectorized distance calculations
    hand_coords = np.array([[hand.landmark[i].x * frame_width, hand.landmark[i].y * frame_height] for hand in hand_landmarks for i in range(len(hand.landmark))])

    # Calculate distances from mouth center to each hand landmark
    distances = np.linalg.norm(hand_coords - mouth_center, axis=1)

    # Check if any distance is within the threshold
    return np.any(distances < (frame_height * proximity_threshold))


def main():
    st.set_page_config(page_title="EAR & MAR Detection", layout="wide")

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
        detect_yawning = st.checkbox('Enable Yawning Detection', value=True)
        detect_smoking = st.checkbox('Check for Smoking', value=True)
        check_seatbelt_option = st.checkbox('Check Seatbelt Status', value=True)
        check_light_conditions_option = st.checkbox('Check Light Conditions', value=True)  
        recognize_expression_option = st.checkbox('Recognize Expressions', value=True)  

    # Thresholds Expander
    with st.sidebar.expander("⚙️ Thresholds", expanded=True):
        st.markdown('<div class="expander-header">Threshold Settings</div>', unsafe_allow_html=True)
        ear_threshold = st.slider("Drowsiness EAR Threshold", min_value=0.1, max_value=0.5, value=0.2, step=0.01)
        mar_yawn_threshold = st.slider("Yawn MAR Threshold", min_value=0.5, max_value=1.0, value=0.6, step=0.01)
        area_threshold = st.slider("Glasses Area Threshold", min_value=50, max_value=300, value=100, step=1)
        aspect_ratio_threshold = st.slider("Glasses Aspect Ratio Threshold", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

    exit_app = st.sidebar.button("Exit Application", key="exit_button")
    if exit_app:
        if st.confirm("Are you sure you want to exit?"):
            st.stop()

    # Thresholds
    ear_threshold = 0.23
    mar_yawn_threshold = 0.4

    # Variables
    blink_count = 0
    previous_ear = 0
    yawn_count = 0
    yawn_detected = False
    yawn_start_time = 0
    last_yawn_time = 0
    last_log_time = time.time()

    ear_values = []
    mar_values = []

    # Create a placeholder for the video feed
    video_placeholder = st.empty()
    # Create a placeholder for the plots
    plot_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    while run:
        success, frame = cap.read()
        if not success:
            st.error("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 450))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        hand_results = hands.process(frame_rgb)

        # Check if the camera is blocked
        if check_camera_blocked_option and check_camera_blocked(results):
            cv2.putText(frame, "Camera Blocked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate EAR and MAR
            avg_ear = calculate_avg_ear(landmarks, frame.shape[1], frame.shape[0])
            avg_mar = calculate_avg_mar(landmarks, frame.shape[1], frame.shape[0])

            # Update EAR and MAR values for plotting
            ear_values.append(avg_ear)
            mar_values.append(avg_mar)

            left_pupil = detect_pupil(landmarks, LEFT_EYE_IDXS, frame.shape[1], frame.shape[0])
            right_pupil = detect_pupil(landmarks, RIGHT_EYE_IDXS, frame.shape[1], frame.shape[0])
            cv2.circle(frame, left_pupil, 5, (0, 255, 255), -1)
            cv2.circle(frame, right_pupil, 5, (0, 255, 255), -1)


            if avg_ear < ear_threshold and previous_ear >= ear_threshold:
                blink_count += 1
            previous_ear = avg_ear

            if detect_yawning:
                yawn_increment, yawn_detected, last_yawn_time = detect_yawn(
                    avg_mar, mar_yawn_threshold, yawn_detected, last_yawn_time
                )
                if yawn_increment:
                    yawn_count += 1

            if time.time() - last_log_time >= 5:
                log_data(blink_count, yawn_count, avg_ear, avg_mar)
                last_log_time = time.time()

            # Update the plot in the placeholder
            with plot_placeholder.container():
                # Clear previous plots
                plt.clf()

                # Create a figure for plotting
                fig, ax = plt.subplots(figsize=(10, 4))

                # Plot EAR
                ax.plot(ear_values, label='Average EAR', color='blue')
                ax.axhline(y=ear_threshold, color='red', linestyle='--', label='Drowsiness Threshold')

                # Plot MAR
                ax.plot(mar_values, label='Average MAR', color='orange')
                ax.axhline(y=mar_yawn_threshold, color='green', linestyle='--', label='Yawn Threshold')

                # Set titles and labels
                ax.set_title('Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR)')
                ax.set_xlabel('Frames')
                ax.set_ylabel('Ratio')
                ax.legend()

                # Display the plot
                st.pyplot(fig)


            # Add text overlay for EAR and MAR
            cv2.putText(frame, f'Avg EAR: {avg_ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Avg MAR: {avg_mar:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Blink Count: {blink_count}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Yawn Count: {yawn_count}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if check_glasses_option:
                glasses_status = check_for_glasses(landmarks, frame, area_threshold, aspect_ratio_threshold)
                cv2.putText(frame, glasses_status, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            if check_seatbelt_option:
                seatbelt_status = check_seatbelt(frame)
                seatbelt_text = "Seatbelt On" if not seatbelt_status else "Seatbelt Off"
                cv2.putText(frame, seatbelt_text, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not seatbelt_status else (0, 0, 255), 2)

            if check_light_conditions_option:
                light_condition_message = check_light_conditions(frame)
                cv2.putText(frame, light_condition_message, (20, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if "Poor light conditions. Please increase lighting." in light_condition_message:
                    # Apply night vision enhancements
                    frame = adjust_brightness_contrast(frame, brightness=30, contrast=30)
                    frame = histogram_equalization(frame)
                    frame = apply_color_map(frame)

            if recognize_expression_option:
                expression = recognize_expression(frame)
                cv2.putText(frame, f'Expression: {expression}', (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if detect_smoking and hand_results.multi_hand_landmarks:
                smoking_detected = is_smoking(landmarks, hand_results.multi_hand_landmarks, frame.shape[1], frame.shape[0])
                if smoking_detected:
                    cv2.putText(frame, 'Smoking Detected!', (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    st.warning("Smoking detected! Please refrain from smoking while driving.")

            if detect_drowsiness and avg_ear < ear_threshold:
                cv2.putText(frame, 'Drowsy Alert!', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                st.warning("Drowsiness detected! Please take a break.")

            if show_landmarks:
                for i in range(len(landmarks)):
                    x = int(landmarks[i].x * frame.shape[1])
                    y = int(landmarks[i].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

                plot_landmarks(frame, landmarks, LEFT_EYE_IDXS)
                plot_landmarks(frame, landmarks, RIGHT_EYE_IDXS)
                plot_mouth_landmarks(frame, landmarks, MOUTH_COORDS)

        # Display the frame
        video_placeholder.image(frame, channels='BGR', use_column_width=True)

        time.sleep(0.1)  # Adjust sleep to control frame rate

    cap.release()

if __name__ == "__main__":
    main()


















