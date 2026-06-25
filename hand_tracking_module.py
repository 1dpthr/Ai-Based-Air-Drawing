import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ABSL_LOGGING_VERBOSITY"] = "0"
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import urllib.request
import os
import time

# Hand detection runs on this width for speed; landmarks map to display size.
DETECT_WIDTH = 480


class HandDetector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        detect_width=DETECT_WIDTH,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.detect_width = detect_width
        self.target_w = 640
        self.target_h = 480

        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print(f"Downloading MediaPipe Hand Landmarker model to {model_path}...")
            url = (
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)
            print("Download complete!")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.running_mode = VisionRunningMode.IMAGE if static_image_mode else VisionRunningMode.VIDEO

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=self.running_mode,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.hands = HandLandmarker.create_from_options(options)
        self.tip_ids = [4, 8, 12, 16, 20]
        self.results = None
        self.frame_count = 0
        self.lm_list = []

        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
        ]

    def set_target_size(self, width, height):
        self.target_w = width
        self.target_h = height

    def _detect_frame(self, img):
        """Run MediaPipe on a downscaled copy for lower latency."""
        h, w = img.shape[:2]
        if w > self.detect_width:
            nw = self.detect_width
            nh = max(1, int(h * nw / w))
            small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:
            small = img

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.running_mode == mp.tasks.vision.RunningMode.VIDEO:
            self.frame_count += 1
            self.results = self.hands.detect_for_video(mp_image, self.frame_count)
        else:
            self.results = self.hands.detect(mp_image)

    def find_hands(self, img, draw=True):
        self._detect_frame(img)
        if draw and self.results and getattr(self.results, "hand_landmarks", None):
            for hand_landmarks in self.results.hand_landmarks:
                self._draw_landmarks(img, hand_landmarks)
        return img

    def _draw_landmarks(self, img, landmarks):
        self._draw_landmarks_styled(img, landmarks)

    def _draw_landmarks_styled(self, img, landmarks, y_min=0, y_max=None, x_margin=0):
        w, h = self.target_w, self.target_h
        if y_max is None:
            y_max = h

        line_color = (210, 85, 195)
        joint_color = (255, 255, 255)
        tip_colors = {4: (0, 200, 255), 8: (80, 230, 80), 12: (255, 180, 80)}

        for start_idx, end_idx in self.HAND_CONNECTIONS:
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            x1, y1 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
            x2, y2 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
            if y_min <= y1 <= y_max and y_min <= y2 <= y_max:
                cv2.line(img, (x1, y1), (x2, y2), line_color, 2)

        for idx, lm in enumerate(landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if not (y_min <= cy <= y_max and x_margin <= cx <= w - x_margin):
                continue
            color = tip_colors.get(idx, joint_color)
            radius = 6 if idx in tip_colors else 3
            cv2.circle(img, (cx, cy), radius, color, -1)

    def draw_hand_overlay(self, img, canvas_top, canvas_bottom, x_margin=10):
        if not self.results or not getattr(self.results, "hand_landmarks", None):
            return False
        for hand_landmarks in self.results.hand_landmarks:
            self._draw_landmarks_styled(
                img, hand_landmarks, y_min=canvas_top, y_max=canvas_bottom, x_margin=x_margin
            )
        return True

    def find_position(self, hand_no=0):
        """Landmark positions mapped to target (display) resolution."""
        self.lm_list = []
        if self.results and getattr(self.results, "hand_landmarks", None):
            if hand_no < len(self.results.hand_landmarks):
                my_hand = self.results.hand_landmarks[hand_no]
                w, h = self.target_w, self.target_h
                for idx, lm in enumerate(my_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([idx, cx, cy])
        return self.lm_list

    def fingers_up(self):
        fingers = []
        if len(self.lm_list) == 0:
            return []

        thumb_tip = self.lm_list[self.tip_ids[0]]
        wrist = self.lm_list[0]
        index_mcp = self.lm_list[5]
        thumb_dist = abs(thumb_tip[1] - wrist[1])
        thumb_extended = thumb_dist > abs(index_mcp[2] - wrist[2]) * 0.6
        fingers.append(1 if thumb_extended else 0)

        for finger_id in range(1, 5):
            tip = self.lm_list[self.tip_ids[finger_id]]
            pip = self.lm_list[self.tip_ids[finger_id] - 2]
            fingers.append(1 if tip[2] < pip[2] - 8 else 0)

        return fingers

    def _finger_extension(self, tip_id, pip_id):
        return self.lm_list[pip_id][2] - self.lm_list[tip_id][2]

    def is_drawing_pose(self):
        if len(self.lm_list) < 21:
            return False

        index_ext = self._finger_extension(8, 6)
        middle_ext = self._finger_extension(12, 10)
        ring_ext = self._finger_extension(16, 14)
        pinky_ext = self._finger_extension(20, 18)

        index_up = index_ext > 18
        others_low = middle_ext < 35 and ring_ext < 35 and pinky_ext < 35
        index_dominant = index_ext > middle_ext + 8

        return index_up and others_low and index_dominant

    def is_selection_pose(self):
        if len(self.lm_list) < 21:
            return False
        fingers = self.fingers_up()
        if len(fingers) < 5:
            return False
        return fingers[1] and fingers[2] and not fingers[3] and not fingers[4]
