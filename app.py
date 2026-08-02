import os
import cv2
import numpy as np
import torch
import joblib
import urllib.request
import bz2
from collections import deque
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from scipy.stats import skew, kurtosis, iqr, entropy
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ==========================================
# 1. FEATURE EXTRACTION & CLASSICAL DESCRIPTORS
# ==========================================
def calculate_hjorth_parameters(a):
    """Calculates Hjorth activity, mobility, and complexity parameters."""
    activity = np.var(a, axis=1)
    diff1 = np.diff(a, axis=1)
    diff2 = np.diff(diff1, axis=1) if a.shape[1] > 2 else np.zeros((a.shape[0], max(0, a.shape[1]-1)))
    
    var_diff1 = np.var(diff1, axis=1) if diff1.shape[1] > 0 else np.zeros_like(activity)
    var_diff2 = np.var(diff2, axis=1) if diff2.shape[1] > 0 else np.zeros_like(activity)
    
    mobility = np.sqrt(var_diff1 / (activity + 1e-8))
    mobility_diff1 = np.sqrt(var_diff2 / (var_diff1 + 1e-8))
    complexity = mobility_diff1 / (mobility + 1e-8)
    return activity, mobility, complexity

def extract_advanced_classical_features(tensor_windows):
    """
    Transforms sequence tensors into an expanded set of comprehensive 
    temporal, frequency, entropy, and domain-specific descriptors.
    """
    N, T, F = tensor_windows.shape
    np_win = tensor_windows.numpy()
    
    # 1. Basic Statistical Moments
    w_mean = np.mean(np_win, axis=1)
    w_median = np.median(np_win, axis=1)
    w_std = np.std(np_win, axis=1)
    w_var = np.var(np_win, axis=1)
    w_min = np.min(np_win, axis=1)
    w_max = np.max(np_win, axis=1)
    w_range = w_max - w_min
    w_iqr = iqr(np_win, axis=1)
    w_rms = np.sqrt(np.mean(np_win**2, axis=1))
    w_energy = np.sum(np_win**2, axis=1)
    w_skew = skew(np_win, axis=1, nan_policy='omit')
    w_kurt = kurtosis(np_win, axis=1, nan_policy='omit')
    w_mad = np.mean(np.abs(np_win - np.mean(np_win, axis=1, keepdims=True)), axis=1)
    
    # 2. Derivatives (Velocity & Acceleration)
    w_diff1 = np.diff(np_win, axis=1)
    w_diff2 = np.diff(w_diff1, axis=1) if T > 2 else np.zeros((N, max(0, T-1), F))
    
    w_diff1_mean = np.mean(w_diff1, axis=1) if w_diff1.shape[1] > 0 else np.zeros((N, F))
    w_diff1_std = np.std(w_diff1, axis=1) if w_diff1.shape[1] > 0 else np.zeros((N, F))
    w_diff2_mean = np.mean(w_diff2, axis=1) if w_diff2.shape[1] > 0 else np.zeros((N, F))
    w_diff2_std = np.std(w_diff2, axis=1) if w_diff2.shape[1] > 0 else np.zeros((N, F))
    
    # 3. Hjorth Parameters
    activity, mobility, complexity = calculate_hjorth_parameters(np_win)
    
    # 4. Zero Crossing Rate
    zcr = np.mean(np.diff(np.signbit(np_win), axis=1), axis=1)
    
    # 5. Frequency Domain Features
    fft_vals = np.fft.rfft(np_win, axis=1)
    fft_power = np.abs(fft_vals)**2
    fft_mean_freq = np.mean(fft_power, axis=1)
    fft_std_freq = np.std(fft_power, axis=1)
    
    # 6. Entropy & Variability Measures
    ent_list = []
    for i in range(N):
        row_ent = []
        for j in range(F):
            hist, _ = np.histogram(np_win[i, :, j], bins=10, density=True)
            row_ent.append(entropy(hist + 1e-9))
        ent_list.append(row_ent)
    w_entropy = np.array(ent_list)

    # 7. Domain Specific: PERCLOS and Blink Dynamics proxies
    perclos = np.mean(np_win < -0.5, axis=1)
    closure_duration = np.sum(np_win < -0.5, axis=1)

    features = np.hstack([
        w_mean, w_median, w_std, w_var, w_min, w_max, w_range, 
        w_iqr, w_rms, w_energy, w_skew, w_kurt, w_mad, 
        w_diff1_mean, w_diff1_std, w_diff2_mean, w_diff2_std,
        activity, mobility, complexity, zcr, 
        fft_mean_freq, fft_std_freq, w_entropy,
        perclos, closure_duration
    ])
    
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


class DlibAdvancedDrowsinessFeatureExtractor:
    def __init__(self, fps=30, window_size=60):
        self.fps = fps
        self.window_size = window_size
        
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            print("Downloading Dlib landmark predictor model...")
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
            with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2") as fr, open(predictor_path, "wb") as fw:
                fw.write(fr.read())
            print("Model downloaded and extracted successfully.")

        import dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        self.prev_avg_ear = None
        self.prev_mar = None
        self.prev_head_center = None
        
        self.pitch_history = deque(maxlen=5)
        self.yaw_history = deque(maxlen=5)
        self.roll_history = deque(maxlen=5)
        
        self.ear_history = deque(maxlen=self.window_size)
        self.is_blinking = False
        self.blink_start_frame = 0
        self.consecutive_closed_frames = 0
        self.min_consecutive_closed_for_blink = 3
        self.current_blink_duration = 0.0
        self.current_closure_duration = 0.0
        self.closure_threshold = 0.22

        self.LEFT_EYE_INDICES = list(range(36, 42))
        self.RIGHT_EYE_INDICES = list(range(42, 48))
        self.MOUTH_INDICES = list(range(48, 68))

        self.model_3d_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

    def _calculate_ear(self, shape, eye_indices, w, h):
        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in eye_indices], dtype=np.float64)
        vertical_dist_1 = np.linalg.norm(pts[1] - pts[5])
        vertical_dist_2 = np.linalg.norm(pts[2] - pts[4])
        horizontal_dist = np.linalg.norm(pts[0] - pts[3])
        ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist + 1e-6)
        return float(ear)

    def _calculate_mar(self, shape, w, h):
        upper_lip = np.array([shape.part(51).x, shape.part(51).y], dtype=np.float64)
        lower_lip = np.array([shape.part(57).x, shape.part(57).y], dtype=np.float64)
        left_corner = np.array([shape.part(48).x, shape.part(48).y], dtype=np.float64)
        right_corner = np.array([shape.part(54).x, shape.part(54).y], dtype=np.float64)
        vertical_dist = np.linalg.norm(upper_lip - lower_lip)
        horizontal_dist = np.linalg.norm(left_corner - right_corner)
        mar = vertical_dist / (horizontal_dist + 1e-6)
        return float(mar)

    def _estimate_head_pose(self, shape, w, h):
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),
            (shape.part(8).x, shape.part(8).y),
            (shape.part(36).x, shape.part(36).y),
            (shape.part(45).x, shape.part(45).y),
            (shape.part(48).x, shape.part(48).y),
            (shape.part(54).x, shape.part(54).y)
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_3d_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        raw_pitch = float(euler_angle[0][0])
        raw_yaw = float(euler_angle[1][0])
        raw_roll = float(euler_angle[2][0])

        self.pitch_history.append(raw_pitch)
        self.yaw_history.append(raw_yaw)
        self.roll_history.append(raw_roll)

        return float(np.mean(self.pitch_history)), float(np.mean(self.yaw_history)), float(np.mean(self.roll_history))

    def extract_frame_metrics(self, img_bgr, frame_idx):
        height, width = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector(gray, 0)
        face_detected = len(faces) > 0

        if not face_detected:
            self.prev_avg_ear = None
            self.prev_mar = None
            self.prev_head_center = None
            self.consecutive_closed_frames = 0
            return [0.0] * 19, False

        rect = faces[0]
        shape = self.predictor(gray, rect)

        cv2.rectangle(img_bgr, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 255), 2)
        for idx in self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES + self.MOUTH_INDICES:
            cv2.circle(img_bgr, (shape.part(idx).x, shape.part(idx).y), 2, (0, 255, 0), -1)

        ear_left = self._calculate_ear(shape, self.LEFT_EYE_INDICES, width, height)
        ear_right = self._calculate_ear(shape, self.RIGHT_EYE_INDICES, width, height)
        avg_ear = float((ear_left + ear_right) / 2.0)
        ear_diff = float(abs(ear_left - ear_right))
        self.ear_history.append(avg_ear)

        mar = self._calculate_mar(shape, width, height)
        pitch, yaw, roll = self._estimate_head_pose(shape, width, height)

        face_width = max(rect.width(), 1)
        face_height = max(rect.height(), 1)
        face_diagonal = np.sqrt(face_width**2 + face_height**2)

        face_center_x = (rect.left() + rect.right()) / 2.0
        face_center_y = (rect.top() + rect.bottom()) / 2.0
        current_center = np.array([face_center_x, face_center_y])
        
        if self.prev_head_center is None:
            head_motion = 0.0
        else:
            head_motion = float(np.linalg.norm(current_center - self.prev_head_center) / face_diagonal)
        self.prev_head_center = current_center

        left_eye_center = np.mean([(shape.part(i).x, shape.part(i).y) for i in self.LEFT_EYE_INDICES], axis=0)
        right_eye_center = np.mean([(shape.part(i).x, shape.part(i).y) for i in self.RIGHT_EYE_INDICES], axis=0)
        
        l_width = max(np.linalg.norm(np.array([shape.part(36).x, shape.part(36).y]) - np.array([shape.part(39).x, shape.part(39).y])), 1.0)
        r_width = max(np.linalg.norm(np.array([shape.part(42).x, shape.part(42).y]) - np.array([shape.part(45).x, shape.part(45).y])), 1.0)

        left_eyecenter_x = float((left_eye_center[0] - shape.part(36).x) / l_width)
        left_eyecenter_y = float((left_eye_center[1] - shape.part(36).y) / l_width)
        right_eyecenter_x = float((right_eye_center[0] - shape.part(42).x) / r_width)
        right_eyecenter_y = float((right_eye_center[1] - shape.part(42).y) / r_width)

        ear_velocity = 0.0 if self.prev_avg_ear is None else float(avg_ear - self.prev_avg_ear)
        mar_velocity = 0.0 if self.prev_mar is None else float(mar - self.prev_mar)
        self.prev_avg_ear = avg_ear
        self.prev_mar = mar

        blink_state = 0.0
        if avg_ear < self.closure_threshold:
            self.consecutive_closed_frames += 1
            if self.consecutive_closed_frames >= self.min_consecutive_closed_for_blink:
                blink_state = 1.0
                if not self.is_blinking:
                    self.is_blinking = True
                    self.blink_start_frame = frame_idx - (self.min_consecutive_closed_for_blink - 1)
                self.current_closure_duration = float((frame_idx - self.blink_start_frame + 1) / self.fps)
                self.current_blink_duration = 0.0
        else:
            self.consecutive_closed_frames = 0
            if self.is_blinking:
                self.current_blink_duration = float((frame_idx - self.blink_start_frame) / self.fps)
                self.is_blinking = False
            else:
                self.current_blink_duration = 0.0
            self.current_closure_duration = 0.0

        if len(self.ear_history) > 0:
            closed_count = sum(1 for val in self.ear_history if val < self.closure_threshold)
            perclos = float(closed_count / len(self.ear_history))
        else:
            perclos = 0.0

        feature_vector = [
            ear_left, ear_right, avg_ear, ear_diff, mar, pitch, yaw, roll,
            left_eyecenter_x, left_eyecenter_y, right_eyecenter_x, right_eyecenter_y,
            ear_velocity, mar_velocity, blink_state, self.current_blink_duration,
            self.current_closure_duration, perclos, head_motion
        ]

        return feature_vector, True


# ==========================================
# 2. STREAMLIT WEBRTC VIDEO PROCESSOR
# ==========================================
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        model_path = "best_drowsiness_model_logistic_regression.pkl"
        if os.path.exists(model_path):
            package = joblib.load(model_path)
            self.scaler = package["scaler"]
            self.selector = package["selector"]
            self.model = package["classifier"]
            self.model_loaded = True
        else:
            self.model_loaded = False

        self.window_size = 60
        self.extractor = DlibAdvancedDrowsinessFeatureExtractor(fps=30, window_size=self.window_size)
        self.frame_buffer = deque(maxlen=self.window_size)
        self.frame_idx = 0
        self.inference_interval = 4
        
        self.last_probability = 0.0
        self.last_status_text = "Collecting Data..."
        self.last_status_color = (255, 255, 0)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_idx += 1

        features, face_detected = self.extractor.extract_frame_metrics(img, self.frame_idx)
        
        if face_detected:
            self.frame_buffer.append(features)
        
        if self.model_loaded and len(self.frame_buffer) == self.window_size and (self.frame_idx % self.inference_interval == 0):
            win_array = np.array(self.frame_buffer, dtype=np.float32)
            tensor_win = torch.tensor(win_array, dtype=torch.float32).unsqueeze(0)
            
            classical_features = extract_advanced_classical_features(tensor_win)
            scaled_features = self.scaler.transform(classical_features)
            selected_features = self.selector.transform(scaled_features)
            
            self.last_probability = self.model.predict_proba(selected_features)[0, 1]
            prediction = self.model.predict(selected_features)[0]
            
            if prediction == 1 or self.last_probability > 0.5:
                self.last_status_text = f"DROWSY ALERT! ({self.last_probability:.2f})"
                self.last_status_color = (0, 0, 255)  # Red
            else:
                self.last_status_text = f"Awake ({self.last_probability:.2f})"
                self.last_status_color = (0, 255, 0)  # Green
        elif not self.model_loaded:
            self.last_status_text = "Model file missing!"
            self.last_status_color = (0, 0, 255)

        # UI Overlay details onto the video frame
        cv2.putText(img, self.last_status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.last_status_color, 2)
        cv2.putText(img, f"Buffer: {len(self.frame_buffer)}/{self.window_size}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==========================================
# 3. STREAMLIT APP UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="Real-Time Drowsiness Detector", layout="centered")

st.title("🚗 Real-Time Driver Drowsiness Detection System")
st.write("Using browser WebRTC streaming, Dlib facial landmarks, and classical ML feature extraction.")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor
)
