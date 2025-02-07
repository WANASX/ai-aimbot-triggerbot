import sys
import json
import os
import time as t
import random as rd
import logging
from collections import deque
import math as m
import copy

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
    QVBoxLayout, QGroupBox, QMessageBox, QDesktopWidget, QDialog, QSlider, QFormLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint

import numpy as np
import keyboard as kb
from screeninfo import get_monitors as gm
from ultralytics import YOLO
import torch as tc
import cv2 as cv
from threading import Thread, Lock
import difflib
from PIL import Image, ImageEnhance
import pytesseract
import win32api
import bettercam
import mss
import warnings
import ctypes
import webbrowser

script_dir = os.path.dirname(os.path.abspath(__file__))
libs_dir = os.path.join(script_dir, 'libs')
CONFIG_PATH = os.path.join(libs_dir, 'config.json')
MODEL_PATH_DEFAULT = os.path.join(libs_dir, 'model1.pt')
PROFILES_PATH = os.path.join(libs_dir, 'profiles.json')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.FileHandler("GamerFunNoLogitech.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants for mouse_event function
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

user32 = ctypes.windll.user32

class CtypesMouse:
    @staticmethod
    def move_relative(dx, dy):
        user32.mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, 0)

    @staticmethod
    def mouse_down(button_flag):
        # button_flag 0x01 for left, 0x02 for right
        if button_flag == 0x01:
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        elif button_flag == 0x02:
            user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)

    @staticmethod
    def mouse_up(button_flag):
        if button_flag == 0x01:
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        elif button_flag == 0x02:
            user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

def load_config(config_file=CONFIG_PATH):
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.warning("Config file not found. Creating default config.")
        default_config = {
            "aimbot": False,
            "use_cuda": False,
            "trigger_bot_enabled": False,
            "trigger_bot_active": False,
            "aim_speed": 0.05,
            "aim_target_choice": "neck",
            "keys": {"activate": "alt", "exit": "insert"},
            "move_mouse": {"steps": 40, "delay": 0.0009},
            "aspect_ratios": {
                "16:9": [
                    [773, 257],
                    [953, 257],
                    [953, 289],
                    [773, 289]
                ],
                "5:3": [
                    [773, 275],
                    [956, 275],
                    [956, 305],
                    [773, 305]
                ],
                "16:10": [
                    [773, 285],
                    [953, 285],
                    [953, 317],
                    [773, 317]
                ],
                "3:2": [
                    [773, 302],
                    [947, 302],
                    [947, 331],
                    [773, 332]
                ],
                "4:3": [
                    [773, 327],
                    [945, 327],
                    [945, 352],
                    [773, 353]
                ],
                "5:4": [
                    [773, 340],
                    [943, 340],
                    [943, 363],
                    [773, 364]
                ],
                "21:9": [
                    [773, 240],
                    [1000, 240],
                    [1000, 270],
                    [773, 270]
                ]
            },
            "selected_aspect_ratio": "16:9",
            "model_path": "model1.pt",
            "recoil_control_enabled": False,
            "agent": "THERMITE"
        }
        save_config(default_config, config_file)
        return default_config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def save_config(config, config_file=CONFIG_PATH):
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info("Configuration saved successfully.")
    except Exception as e:
        logger.error(f"Error saving config: {e}")

def load_coordinates(config_file=CONFIG_PATH):
    try:
        config = load_config(config_file)
        aspect_ratio = config.get('selected_aspect_ratio', '16:9')
        coordinates = np.array(config['aspect_ratios'][aspect_ratio])
        logger.info(f"Loaded coordinates for aspect ratio {aspect_ratio}.")
        return coordinates, aspect_ratio
    except Exception as e:
        logger.error(f"Error loading coordinates: {e}")
        return (None, None)

def load_profiles(profiles_path=PROFILES_PATH):
    external_profiles_path = profiles_path
    try:
        with open(external_profiles_path, 'r') as profiles_file:
            profiles = json.load(profiles_file)
        profiles_dict = {profile["profile"].upper(): profile for profile in profiles}
        logger.info(f"Loaded {len(profiles_dict)} recoil profiles.")
        return profiles_dict
    except FileNotFoundError:
        logger.error(f"Profiles file not found at {external_profiles_path}.")
        return {}
    except Exception as e:
        logger.error(f"Error loading profiles: {e}")
        return {}

class Communicate(QObject):
    config_updated = pyqtSignal()
    aimbot_toggled = pyqtSignal(bool)
    trigger_bot_toggled = pyqtSignal(bool)
    trigger_bot_active_toggled = pyqtSignal(bool)
    recoil_control_toggled = pyqtSignal(bool)
    agent_detect_requested = pyqtSignal()
    agent_selected_manually = pyqtSignal(str)
    recoil_strength_updated = pyqtSignal(float)

class Aimbot(Thread):
    def __init__(self, config, comm):
        super().__init__(daemon=True)
        self.config = config
        self.comm = comm
        self.trigger_bot_enabled = self.config.get("trigger_bot_enabled", False)
        self.trigger_bot_active = self.config.get("trigger_bot_active", False)
        self.running = False
        self.lock = Lock()
        self.latest_frame = None
        self.model = None
        self.monitor = None
        self.target_off = self.config.get("aim_targets", {}).get(self.config.get("aim_target_choice", "neck"), 0.2)
        self.activation_key = self.config.get("keys", {}).get("activate", "alt")
        self.load_dependencies()
        self.capture_times = deque(maxlen=100)
        self.inference_times = deque(maxlen=100)
        self.last_log_time = t.time()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    def load_dependencies(self):
        try:
            self.device = 'cuda' if self.config.get("use_cuda", False) and tc.cuda.is_available() else 'cpu'
            model_filename = self.config.get("model_path", "model1.pt")
            model_path = os.path.join(self.script_dir, 'libs', model_filename)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.model = YOLO(model_path).to(self.device)
            if self.device == 'cuda':
                self.model.fuse()
                self.model.half()
                logger.info("Aimbot model fused and converted to half precision for CUDA.")
            self.model.eval()
            logger.info(f"Aimbot model loaded on {self.device}.")
        except Exception as e:
            logger.error(f"Error loading Aimbot model: {e}")
            self.model = None

        try:
            monitor_info = gm()[0]
            size_region = self.config.get("resolution", {}).get("width", 320)
            cross_x, cross_y = monitor_info.width // 2, monitor_info.height // 2
            self.monitor = {
                "left": cross_x - size_region // 2,
                "top": cross_y - size_region // 2,
                "width": size_region,
                "height": size_region
            }
            logger.info(f"Aimbot screen capture region set to: {self.monitor}")
        except Exception as e:
            logger.error(f"Error setting Aimbot screen capture region: {e}")
            self.monitor = None

        self.frame_skip_range = [1, 1]
        self.frame_count = 0
        self.offset_range = self.config.get("randomization", {}).get("small_offset", [0, 0])
        self.click_delay_range = self.config.get("randomization", {}).get("click_delay_range", [0.01, 0.03])
        self.sleep_range = self.config.get("randomization", {}).get("cpu_sleep_range", [0.001, 0.003])
        self.aim_spd = self.config.get("aim_speed", 0.05)
        self.move_steps = self.config.get("move_mouse", {}).get("steps", 40)
        self.move_delay = self.config.get("move_mouse", {}).get("delay", 0.0009)

    def run(self):
        if not self.model or not self.monitor:
            logger.warning("Aimbot not properly initialized. Exiting thread.")
            return
        self.running = True
        logger.info("Aimbot thread started.")
        capture_thread = Thread(target=self.screen_capture, daemon=True)
        action_thread = Thread(target=self.inference_and_action, daemon=True)
        capture_thread.start()
        action_thread.start()
        while self.running:
            t.sleep(1)

    def screen_capture(self):
        with mss.mss() as sc:
            last_capture_time = 0
            capture_interval = 0.05
            while self.running:
                try:
                    current_time = t.time()
                    if (current_time - last_capture_time) >= capture_interval:
                        if kb.is_pressed(self.activation_key) and self.config.get("aimbot", False):
                            screenshot = sc.grab(self.monitor)
                            frame = np.array(screenshot)[:, :, :3]
                            with self.lock:
                                self.latest_frame = frame
                                self.capture_times.append(current_time)
                            last_capture_time = current_time
                    t.sleep(0.001)
                except Exception as e:
                    logger.error(f"Aimbot Error during screen capture: {e}")

    def inference_and_action(self):
        while self.running:
            try:
                with self.lock:
                    frame = self.latest_frame
                    self.latest_frame = None
                if (frame is not None and
                    kb.is_pressed(self.activation_key) and
                    self.config.get("aimbot", False)):
                    preprocessed_frame = self.preprocess_frame(frame)
                    if preprocessed_frame is None:
                        continue
                    start_inference = t.time()
                    results = self.model.predict(
                        source=preprocessed_frame,
                        device=self.device,
                        conf=self.config.get("conf_threshold", 0.4),
                        iou=self.config.get("iou_threshold", 0.45),
                        verbose=False
                    )
                    end_inference = t.time()
                    inference_time = end_inference - start_inference
                    self.inference_times.append(inference_time)
                    current_time = t.time()
                    if (current_time - self.last_log_time) >= 1.0:
                        while self.capture_times and (current_time - self.capture_times[0]) > 1.0:
                            self.capture_times.popleft()
                        capture_fps = len(self.capture_times)
                        process_fps = len(self.inference_times)
                        avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0.0
                        logger.info(f"Aimbot: Capture FPS: {capture_fps}, Process FPS: {process_fps}")
                        logger.info(f"Aimbot: Average Inference Time: {avg_inference_time:.4f} seconds")
                        self.last_log_time = current_time
                    found_target = False
                    for result in results:
                        for box in result.boxes:
                            try:
                                confidence = box.conf.item()
                                class_id = int(box.cls.item())
                                if confidence < 0.4 or class_id != 0:
                                    continue
                                box_xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
                                if len(box_xyxy) < 4:
                                    logger.error(f"Box data is incomplete: {box_xyxy}")
                                    continue
                                x1, y1, x2, y2 = box_xyxy[:4]
                                target_y = int(y1 + (y2 - y1) * self.target_off)
                                target_x = (x1 + x2) // 2
                                offset_x = target_x - self.monitor["width"] // 2
                                offset_y = target_y - self.monitor["height"] // 2
                                self.smooth_move(offset_x, offset_y)
                                if x1 <= self.monitor["width"] // 2 <= x2 and y1 <= self.monitor["height"] // 2 <= y2:
                                    self.trigger_shot()
                                    found_target = True
                                    break
                            except Exception as inner_e:
                                logger.error(f"Error processing a single detection: {inner_e}")
                                continue
                        if found_target:
                            break
            except Exception as e:
                logger.error(f"Aimbot Error during inference and action: {e}")
            t.sleep(0.001)

    def preprocess_frame(self, frame):
        try:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            return rgb_frame
        except Exception as e:
            logger.error(f"Aimbot Error during frame preprocessing: {e}")
            return None

    def mouse_move(self, dx, dy):
        try:
            CtypesMouse.move_relative(dx, dy)
        except Exception as e:
            logger.error(f"Aimbot Error moving mouse: {e}")

    def smooth_move(self, dx, dy):
        try:
            dist = m.hypot(dx, dy)
            if dist == 0:
                return
            base_steps = self.move_steps
            base_delay = self.move_delay
            aim_speed = self.aim_spd
            steps = max(1, int(base_steps / aim_speed))
            delay = base_delay / aim_speed
            step_x, step_y = dx / steps, dy / steps
            for i in range(steps):
                if not self.running:
                    break
                move_x = int(step_x)
                move_y = int(step_y)
                self.mouse_move(move_x, move_y)
                t.sleep(delay)
        except Exception as e:
            logger.error(f"Aimbot Error during smooth mouse movement: {e}")

    def trigger_shot(self):
        if self.trigger_bot_enabled and self.trigger_bot_active:
            try:
                CtypesMouse.mouse_down(0x01)
                t.sleep(rd.uniform(*self.click_delay_range))
                CtypesMouse.mouse_up(0x01)
            except Exception as e:
                logger.error(f"Aimbot Error executing left-click: {e}")

    def update_config(self, new_config):
        with self.lock:
            use_cuda_prev = self.config.get("use_cuda", False)
            model_path_prev = self.config.get("model_path", "model1.pt")

            self.config.update(new_config)

            self.trigger_bot_enabled = self.config.get("trigger_bot_enabled", False)
            self.trigger_bot_active = self.config.get("trigger_bot_active", False)
            self.activation_key = self.config.get("keys", {}).get("activate", "alt")
            self.aim_spd = self.config.get("aim_speed", 0.05)
            self.move_steps = self.config.get("move_mouse", {}).get("steps", 40)
            self.move_delay = self.config.get("move_mouse", {}).get("delay", 0.0009)

            use_cuda_new = self.config.get("use_cuda", False)
            model_path_new = self.config.get("model_path", "model1.pt")

            if use_cuda_new != use_cuda_prev or model_path_new != model_path_prev:
                logger.info("Aimbot: Detected changes in 'use_cuda' or 'model_path'. Reloading dependencies.")
                self.load_dependencies()

    def stop(self):
        self.running = False
        logger.info("Aimbot: Stopping thread.")
class RecoilControl(Thread):
    def __init__(self, config, comm):
        super().__init__(daemon=True)
        self.config = config
        self.comm = comm
        self.recoil_control_enabled = self.config.get("recoil_control_enabled", False)
        self.profile_data = None
        self.running = False
        self.lock = Lock()
        self.recoil_active = False
        self.profiles = load_profiles()

    def load_default_profile(self):
        default_agent = self.config.get("agent", "THERMITE").upper()
        self.set_profile(default_agent)

    def run(self):
        self.running = True
        logger.info("RecoilControl thread started.")
        if self.recoil_control_enabled:
            self.load_default_profile()
        while self.running:
            try:
                if self.recoil_control_enabled and self.profile_data:
                    left_pressed = win32api.GetKeyState(0x01) < 0
                    right_pressed = win32api.GetKeyState(0x02) < 0
                    if left_pressed and right_pressed and not self.recoil_active:
                        self.recoil_active = True
                        start_time = t.time()
                        recoil_duration = self.profile_data.get("timing", 0) * 0.1
                        while (t.time() - start_time) < recoil_duration:
                            if not (win32api.GetKeyState(0x01) < 0 and win32api.GetKeyState(0x02) < 0):
                                break
                            jitter_x = rd.uniform(-0.5, 0.5)
                            jitter_y = rd.uniform(-1, 1)
                            move_x = self.profile_data.get("x", 0) + jitter_x
                            move_y = self.profile_data.get("y", 0) + jitter_y
                            try:
                                CtypesMouse.move_relative(round(move_x), round(move_y))
                            except Exception as e:
                                logger.error(f"RecoilControl: Error moving mouse for recoil: {e}")
                            t.sleep(0.01)
                        self.recoil_active = False
            except Exception as e:
                logger.error(f"RecoilControl: Error in thread: {e}")
            t.sleep(0.05)

    def set_profile(self, agent_name):
        with self.lock:
            agent_name_upper = agent_name.upper()
            profile = self.profiles.get(agent_name_upper, None)
            if profile:
                self.profile_data = profile
                self.recoil_control_enabled = True
                logger.info(f"RecoilControl: Profile '{agent_name}' loaded. Recoil control enabled.")
            else:
                self.profile_data = None
                self.recoil_control_enabled = False
                logger.warning(f"RecoilControl: No profile found for agent '{agent_name}', recoil control disabled.")

    def update_config(self, new_config):
        with self.lock:
            agent_prev = self.config.get('agent', 'THERMITE').upper()
            self.config.update(new_config)
            agent_new = self.config.get('agent', 'THERMITE').upper()

            self.recoil_control_enabled = self.config.get("recoil_control_enabled", False)

            if self.recoil_control_enabled and (agent_new != agent_prev):
                logger.info("RecoilControl: Detected change in 'agent'. Updating profile.")
                self.profiles = load_profiles()
                self.set_profile(agent_new)

    def stop(self):
        self.running = False
        logger.info("RecoilControl: Stopping thread.")

class HotkeyHandler(Thread):
    def __init__(self, comm, config):
        super().__init__(daemon=True)
        self.comm = comm
        self.config = config
        self.running = True

    def run(self):
        try:
            kb.add_hotkey('F5', self.on_f5_pressed)
            kb.add_hotkey('caps lock', self.on_caps_lock_pressed)
            while self.running:
                t.sleep(0.1)
        except Exception as e:
            logger.error(f"HotkeyHandler: Error in thread: {e}")

    def on_f5_pressed(self):
        self.comm.agent_detect_requested.emit()

    def on_caps_lock_pressed(self):
        if not self.config.get("trigger_bot_enabled", False):
            return
        current_state = self.config.get("trigger_bot_active", False)
        new_state = not current_state
        self.comm.trigger_bot_active_toggled.emit(new_state)
        self.config["trigger_bot_active"] = new_state
        save_config(self.config)
        logger.info(f"HotkeyHandler: Trigger Bot active state toggled to {new_state} via Caps Lock.")

    def stop(self):
        self.running = False
        try:
            kb.remove_hotkey('F5')
            kb.remove_hotkey('caps lock')
            logger.info("HotkeyHandler: Hotkeys 'F5' and 'Caps Lock' unbound successfully.")
        except Exception as e:
            logger.error(f"HotkeyHandler: Error unbinding hotkeys: {e}")

class AgentDetectionWindow(QMainWindow):
    agent_detected_signal = pyqtSignal(str)

    def __init__(self, bbox_coords, profiles):
        super().__init__()
        self.setWindowTitle("Agent Detection")
        self.setFixedSize(330, 120)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.bbox_coords = bbox_coords
        self.profiles = profiles
        self.detected_agent = None
        self.init_ui()
        self.position_window_top_center()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self.sct = bettercam.create(output_color="BGR")
                logger.info("AgentDetectionWindow: Initialized bettercam for screen capture.")
        except Exception as e:
            logger.error(f"AgentDetectionWindow: Error initializing bettercam: {e}")
            self.sct = None
        self.scan_enabled = True
        self.scan_timer = QTimer()
        self.scan_timer.timeout.connect(self.capture_and_detect)
        self.scan_timer.start(100)

    def init_ui(self):
        layout = QVBoxLayout()
        self.detected_label = QLabel("Detected Agent: None")
        self.detected_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.detected_label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def position_window_top_center(self):
        screen_geometry = QDesktopWidget().screenGeometry()
        screen_width = screen_geometry.width()
        x = (screen_width - self.width()) // 2
        y = 30
        self.move(x, y)

    def capture_and_detect(self):
        if not self.scan_enabled:
            return
        try:
            if not self.sct:
                logger.error("AgentDetectionWindow: Screen capture device not initialized.")
                return
            screen = self.sct.grab()
            img = np.array(screen)
            if img is None:
                return
            x_min, y_min = self.bbox_coords[0]
            x_max, y_max = self.bbox_coords[2]
            if any(coord < 0 for coord in [x_min, y_min, x_max, y_max]):
                logger.error(f"AgentDetectionWindow: Invalid capture coordinates: {self.bbox_coords}")
                return
            height, width = img.shape[:2]
            if x_max >= width or y_max >= height:
                logger.error(f"AgentDetectionWindow: Capture region exceeds screen bounds: {self.bbox_coords}")
                return
            roi = img[y_min:y_max, x_min:x_max]
            pil_image = Image.fromarray(cv.cvtColor(roi, cv.COLOR_BGR2RGB))
            enhanced_image = ImageEnhance.Contrast(pil_image).enhance(4).convert('L')
            bw_image = enhanced_image.point(lambda x: 255 if x > 128 else 0, mode='1').resize(
                (roi.shape[1] * 2, roi.shape[0] * 2), Image.LANCZOS)
            detected_text_raw = pytesseract.image_to_string(
                bw_image, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ').strip().upper()
            logger.debug(f"AgentDetectionWindow: Detected text (raw): '{detected_text_raw}'")
            if detected_text_raw in self.profiles:
                detected_text = detected_text_raw
            elif len(detected_text_raw) <= 2:
                detected_text = detected_text_raw
            else:
                closest_matches = difflib.get_close_matches(
                    detected_text_raw, list(self.profiles.keys()), n=1, cutoff=0.7)
                detected_text = closest_matches[0] if closest_matches else None
            if detected_text:
                if detected_text != "NONE":
                    self.detected_label.setText(f"Detected Agent: {detected_text}")
                    logger.info(f"AgentDetectionWindow: Detected Agent: {detected_text}")
                    self.agent_detected_signal.emit(detected_text)
                    self.detected_agent = detected_text
                    self.scan_enabled = False
                    self.scan_timer.stop()
                    self.start_countdown()
            else:
                logger.warning(f"AgentDetectionWindow: No close match found for detected text: '{detected_text_raw}'")
                self.detected_label.setText("Detected Agent: None")
        except Exception as e:
            logger.error(f"AgentDetectionWindow: Error during capture and detect: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred during detection:\n{e}")

    def start_countdown(self):
        self.remaining_time = 5
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_timer.start(1000)

    def update_countdown(self):
        self.detected_label.setText(f"Detected Agent: {self.detected_agent} - Closing in: {self.remaining_time} seconds")
        self.remaining_time -= 1
        if self.remaining_time <= 0:
            self.countdown_timer.stop()
            self.close()

    def closeEvent(self, event):
        super().closeEvent(event)
        if self.sct:
            self.sct.release()

class RecoilStrengthEditor(QDialog):
    def __init__(self, current_y, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Recoil Strength (Y)")
        self.setFixedSize(300, 150)
        self.new_y = current_y
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 50)
        self.slider.setValue(int(self.new_y))
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.slider_changed)
        self.value_label = QLabel(f"Y Value: {self.new_y:.1f}")
        layout.addWidget(self.slider)
        layout.addWidget(self.value_label)
        buttons_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save)
        cancel_button = QPushButton("Close")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addStretch()
        buttons_layout.addWidget(save_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def slider_changed(self, value):
        self.new_y = float(value)
        self.value_label.setText(f"Y Value: {self.new_y:.1f}")

    def save(self):
        self.accept()

class ConfigEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Config Editor")
        self.setFixedSize(500, 750)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.offset = QPoint()
        self.config = load_config()
        self.edited_config = copy.deepcopy(self.config)
        self.comm = Communicate()
        self.comm.config_updated.connect(self.on_config_updated)
        self.comm.aimbot_toggled.connect(self.on_aimbot_toggled)
        self.comm.trigger_bot_toggled.connect(self.on_trigger_bot_toggled)
        self.comm.trigger_bot_active_toggled.connect(self.on_trigger_bot_active_toggled)
        self.comm.recoil_control_toggled.connect(self.on_recoil_control_toggled)
        self.comm.agent_detect_requested.connect(self.open_agent_detection)
        self.comm.agent_selected_manually.connect(self.on_agent_selected_manually)
        self.comm.recoil_strength_updated.connect(self.on_recoil_strength_updated)

        self.aimbot = None
        self.recoil_control = None
        self.hotkey_handler = None
        self.agent_window = None
        if self.edited_config.get("aimbot", False):
            self.start_aimbot()
        if self.edited_config.get("recoil_control_enabled", False):
            self.start_recoil_control()
        self.init_ui()
        self.setGeometry(100, 100, 500, 750)

    def init_ui(self):
        main_layout = QVBoxLayout()

        drag_area = QLabel()
        drag_area.setFixedHeight(30)
        drag_area.setStyleSheet("background-color: #2b2b2b;")
        main_layout.addWidget(drag_area)

        aimbot_group = QGroupBox("Aimbot Settings")
        aimbot_layout = QVBoxLayout()

        self.aimbot_enabled = QCheckBox("Enable Aimbot")
        self.aimbot_enabled.setChecked(self.edited_config.get("aimbot", False))
        self.aimbot_enabled.stateChanged.connect(self.toggle_aimbot)
        aimbot_layout.addWidget(self.aimbot_enabled)

        self.use_cuda = QCheckBox("Use CUDA")
        self.use_cuda.setChecked(self.edited_config.get("use_cuda", False))
        self.use_cuda.stateChanged.connect(self.toggle_cuda)
        aimbot_layout.addWidget(self.use_cuda)

        self.trigger_bot_checkbox = QCheckBox("Trigger Bot Enabled")
        self.trigger_bot_checkbox.setChecked(self.edited_config.get("trigger_bot_enabled", False))
        self.trigger_bot_checkbox.stateChanged.connect(self.toggle_trigger_bot)
        aimbot_layout.addWidget(self.trigger_bot_checkbox)

        self.trigger_bot_info_label = QLabel("(To toggle Trigger Bot on/off during the game, press Caps Lock.)")
        self.trigger_bot_info_label.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        aimbot_layout.addWidget(self.trigger_bot_info_label)

        aim_speed_layout = QHBoxLayout()
        aim_speed_label = QLabel("Aim Speed:")
        self.aim_speed = QDoubleSpinBox()
        self.aim_speed.setRange(0.0, 100.0)
        self.aim_speed.setSingleStep(0.1)
        self.aim_speed.setValue(self.edited_config.get("aim_speed", 0.05))
        self.aim_speed.valueChanged.connect(self.change_aim_speed)
        aim_speed_layout.addWidget(aim_speed_label)
        aim_speed_layout.addWidget(self.aim_speed)
        aimbot_layout.addLayout(aim_speed_layout)

        frame_skip_layout = QHBoxLayout()
        frame_skip_label = QLabel("Frame Skip Range:")
        self.frame_skip_min = QSpinBox()
        self.frame_skip_min.setValue(1)
        self.frame_skip_min.setEnabled(False)
        self.frame_skip_max = QSpinBox()
        self.frame_skip_max.setValue(1)
        self.frame_skip_max.setEnabled(False)
        frame_skip_layout.addWidget(frame_skip_label)
        frame_skip_layout.addWidget(self.frame_skip_min)
        frame_skip_layout.addWidget(QLabel("to"))
        frame_skip_layout.addWidget(self.frame_skip_max)
        aimbot_layout.addLayout(frame_skip_layout)

        aim_targets_group = QGroupBox("Aim Targets")
        aim_targets_layout = QVBoxLayout()
        self.aim_target_choice = QComboBox()
        self.aim_target_choice.addItems(["head", "neck", "chest", "legs", "balls"])
        self.aim_target_choice.setCurrentText(self.edited_config.get("aim_target_choice", "neck"))
        self.aim_target_choice.currentTextChanged.connect(self.change_aim_target)
        aim_targets_layout.addWidget(QLabel("Choose Aim Target:"))
        aim_targets_layout.addWidget(self.aim_target_choice)
        aim_targets_group.setLayout(aim_targets_layout)
        aimbot_layout.addWidget(aim_targets_group)

        aimbot_key_layout = QHBoxLayout()
        aimbot_key_label = QLabel("Aimbot Activation Key:")
        self.aimbot_key_selector = QComboBox()
        special_keys = [
            "alt", "shift", "ctrl", "tab", "caps lock", "space", "win",
            "menu", "left ctrl", "right ctrl", "left shift", "right shift"
        ]
        self.aimbot_key_selector.addItems(special_keys)
        current_key = self.edited_config.get("keys", {}).get("activate", "alt")
        index = self.aimbot_key_selector.findText(current_key.lower())
        if index != -1:
            self.aimbot_key_selector.setCurrentIndex(index)
        self.aimbot_key_selector.currentTextChanged.connect(self.change_aimbot_key)
        aimbot_key_layout.addWidget(aimbot_key_label)
        aimbot_key_layout.addWidget(self.aimbot_key_selector)
        aimbot_layout.addLayout(aimbot_key_layout)

        move_mouse_group = QGroupBox("Move Mouse Speed")
        move_mouse_layout = QGridLayout()
        self.move_mouse_steps = QSpinBox()
        self.move_mouse_steps.setValue(self.edited_config.get("move_mouse", {}).get("steps", 40))
        self.move_mouse_steps.setMinimum(1)
        self.move_mouse_steps.valueChanged.connect(self.change_move_mouse_steps)
        self.move_mouse_delay = QDoubleSpinBox()
        self.move_mouse_delay.setDecimals(4)
        self.move_mouse_delay.setRange(0.0001, 1.0)
        self.move_mouse_delay.setValue(self.edited_config.get("move_mouse", {}).get("delay", 0.0009))
        self.move_mouse_delay.setSingleStep(0.0001)
        self.move_mouse_delay.valueChanged.connect(self.change_move_mouse_delay)
        move_mouse_layout.addWidget(QLabel("Steps:"), 0, 0)
        move_mouse_layout.addWidget(self.move_mouse_steps, 0, 1)
        move_mouse_layout.addWidget(QLabel("Delay (s):"), 1, 0)
        move_mouse_layout.addWidget(self.move_mouse_delay, 1, 1)
        move_mouse_group.setLayout(move_mouse_layout)
        aimbot_layout.addWidget(move_mouse_group)

        aimbot_group.setLayout(aimbot_layout)
        main_layout.addWidget(aimbot_group)

        recoil_control_layout = QVBoxLayout()
        recoil_control_container = QGroupBox("Recoil Control Settings")
        recoil_control_inner_layout = QVBoxLayout()

        self.recoil_control_enabled = QCheckBox("Recoil Controller Enabled")
        self.recoil_control_enabled.setChecked(self.edited_config.get("recoil_control_enabled", False))
        self.recoil_control_enabled.stateChanged.connect(self.toggle_recoil_control)
        recoil_control_inner_layout.addWidget(self.recoil_control_enabled)

        self.auto_detection_info = QLabel("(For Auto Detection Press F5 On Operators Screen)")
        self.auto_detection_info.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        recoil_control_inner_layout.addWidget(self.auto_detection_info)

        self.select_agent_manual = QComboBox()
        self.select_agent_manual.addItem("Select Agent Manually")
        self.select_agent_manual.addItems(self.get_all_agents())
        self.select_agent_manual.currentIndexChanged.connect(self.select_agent_manually)
        self.select_agent_manual.setEnabled(self.edited_config.get("recoil_control_enabled", False))
        recoil_control_inner_layout.addWidget(self.select_agent_manual)

        self.edit_recoil_strength_button = QPushButton("Edit Recoil Strength")
        self.edit_recoil_strength_button.clicked.connect(self.edit_recoil_strength)
        self.edit_recoil_strength_button.setEnabled(self.edited_config.get("recoil_control_enabled", False))
        recoil_control_inner_layout.addWidget(self.edit_recoil_strength_button)

        recoil_control_container.setLayout(recoil_control_inner_layout)
        recoil_control_layout.addWidget(recoil_control_container)
        main_layout.addLayout(recoil_control_layout)

        aspect_ratio_layout = QHBoxLayout()
        aspect_ratio_layout.addWidget(QLabel("Selected Aspect Ratio:"))
        self.selected_aspect_ratio = QComboBox()
        self.selected_aspect_ratio.addItems(self.edited_config.get("aspect_ratios", {}).keys())
        self.selected_aspect_ratio.setCurrentText(self.edited_config.get("selected_aspect_ratio", "16:9"))
        self.selected_aspect_ratio.currentTextChanged.connect(self.change_aspect_ratio)
        aspect_ratio_layout.addWidget(self.selected_aspect_ratio)
        main_layout.addLayout(aspect_ratio_layout)

        model_path_layout = QHBoxLayout()
        model_path_label = QLabel("Model Path:")
        self.model_path = QLineEdit(self.edited_config.get("model_path", ""))
        self.model_path.textChanged.connect(self.change_model_path)
        model_browse_button = QPushButton("Browse")
        model_browse_button.clicked.connect(self.browse_model_path)
        model_path_layout.addWidget(model_path_label)
        model_path_layout.addWidget(self.model_path)
        model_path_layout.addWidget(model_browse_button)
        main_layout.addLayout(model_path_layout)

        buttons_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.setMinimumSize(100, 40)
        save_button.clicked.connect(self.save_config)

        cancel_button = QPushButton("Cancel")
        cancel_button.setMinimumSize(100, 40)
        cancel_button.clicked.connect(self.cancel_changes)
        discord_button = QPushButton("Discord")
        discord_button.setMinimumSize(100, 40)
        discord_button.clicked.connect(lambda: webbrowser.open("https://discord.gg/cvVvFrf"))

        # Apply Discord Colors to the Button
        discord_button.setStyleSheet("""
            QPushButton {
                background-color: #5865F2;  /* Discord blue */
                color: #FFFFFF;  /* White text */
                border: none;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4752C4;  /* Slightly darker on hover */
            }
            QPushButton:pressed {
                background-color: #3C45A5;  /* Even darker when pressed */
            }
        """)
        buttons_layout.addWidget(discord_button)

        buttons_layout.addStretch()
        buttons_layout.addWidget(save_button)
        buttons_layout.addWidget(cancel_button)
        main_layout.addLayout(buttons_layout)

        footer_layout = QHBoxLayout()
        footer_message = QLabel("This is a free menu made by wanasx on GamerFun.Club")
        footer_message.setStyleSheet("color: #32CD32; font-weight: bold;")
        footer_message.setAlignment(Qt.AlignCenter)
        footer_layout.addStretch()
        footer_layout.addWidget(footer_message)
        footer_layout.addStretch()
        main_layout.addLayout(footer_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.drag_area = drag_area
        drag_area.mousePressEvent = self.mouse_press_event
        drag_area.mouseMoveEvent = self.mouse_move_event

    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.offset = event.globalPos() - self.frameGeometry().topLeft()

    def mouse_move_event(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.offset)

    def toggle_aimbot(self, state):
        enabled = state == Qt.Checked
        self.edited_config["aimbot"] = enabled
        if enabled and not self.aimbot:
            self.start_aimbot()
        elif not enabled and self.aimbot:
            self.stop_aimbot()
        self.comm.aimbot_toggled.emit(enabled)
        logger.info(f"ConfigEditor: Aimbot Enabled: {enabled}")

    def start_aimbot(self):
        if not self.aimbot or not self.aimbot.is_alive():
            self.aimbot = Aimbot(self.edited_config, self.comm)
            self.aimbot.start()
            logger.info("ConfigEditor: Aimbot thread started.")

    def stop_aimbot(self):
        if self.aimbot and self.aimbot.is_alive():
            self.aimbot.stop()
            self.aimbot.join()
            self.aimbot = None
            logger.info("ConfigEditor: Aimbot thread stopped.")

    def toggle_cuda(self, state):
        use_cuda = state == Qt.Checked
        self.edited_config["use_cuda"] = use_cuda
        if self.aimbot:
            self.aimbot.update_config(self.edited_config)
        logger.info(f"ConfigEditor: Use CUDA: {use_cuda}")

    def change_aim_speed(self, value):
        self.edited_config["aim_speed"] = value
        if self.aimbot:
            self.aimbot.update_config(self.edited_config)
        logger.info(f"ConfigEditor: Aim Speed set to: {value}")

    def change_aim_target(self, value):
        self.edited_config["aim_target_choice"] = value
        if self.aimbot:
            self.aimbot.update_config(self.edited_config)
        logger.info(f"ConfigEditor: Aim Target Choice set to: {value}")

    def toggle_trigger_bot(self, state):
        enabled = state == Qt.Checked
        self.edited_config["trigger_bot_enabled"] = enabled
        if self.aimbot:
            self.aimbot.update_config(self.edited_config)
        self.comm.trigger_bot_toggled.emit(enabled)
        logger.info(f"ConfigEditor: Trigger Bot Enabled: {enabled}")

    def toggle_recoil_control(self, state):
        enabled = state == Qt.Checked
        self.edited_config["recoil_control_enabled"] = enabled
        if enabled and not self.recoil_control:
            self.start_recoil_control()
            self.select_agent_manual.setEnabled(True)
            self.edit_recoil_strength_button.setEnabled(True)
        elif not enabled and self.recoil_control:
            self.stop_recoil_control()
            self.select_agent_manual.setEnabled(False)
            self.edit_recoil_strength_button.setEnabled(False)
        self.comm.recoil_control_toggled.emit(enabled)
        logger.info(f"ConfigEditor: Recoil Control Enabled: {enabled}")

    def start_recoil_control(self):
        if not self.recoil_control or not self.recoil_control.is_alive():
            self.recoil_control = RecoilControl(self.edited_config, self.comm)
            self.recoil_control.start()
            logger.info("ConfigEditor: RecoilControl thread started.")
            if not self.hotkey_handler:
                self.hotkey_handler = HotkeyHandler(self.comm, self.edited_config)
                self.hotkey_handler.start()
                logger.info("ConfigEditor: HotkeyHandler thread started for F5 and Caps Lock.")

    def stop_recoil_control(self):
        if self.recoil_control and self.recoil_control.is_alive():
            self.recoil_control.stop()
            self.recoil_control.join()
            self.recoil_control = None
            logger.info("ConfigEditor: RecoilControl thread stopped.")
        if self.hotkey_handler:
            self.hotkey_handler.stop()
            self.hotkey_handler = None
            logger.info("ConfigEditor: HotkeyHandler thread stopped.")

    def open_agent_detection(self):
        try:
            bbox_coords, selected_aspect_ratio = load_coordinates()
            if bbox_coords is None:
                logger.error("ConfigEditor: Failed to load coordinates.")
                QMessageBox.critical(self, "Error", "Failed to load coordinates from config.")
                return
            profiles_dict = load_profiles()
            if not profiles_dict:
                logger.error("ConfigEditor: No profiles loaded. Please check profiles.json.")
                QMessageBox.critical(self, "Error", "No profiles loaded. Please check profiles.json.")
                return
            self.agent_window = AgentDetectionWindow(bbox_coords, profiles_dict)
            self.agent_window.agent_detected_signal.connect(self.on_agent_detected)
            self.agent_window.show()
            logger.info("ConfigEditor: Agent Detection Window opened.")
        except Exception as e:
            logger.error(f"ConfigEditor: Error opening Agent Detection Window: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred while opening the Agent Detection Window:\n{e}")

    def on_agent_detected(self, agent_name):
        if self.recoil_control:
            self.recoil_control.set_profile(agent_name)
            if self.recoil_control.profile_data:
                logger.info(f"ConfigEditor: Recoil Control activated for agent: {agent_name}")
                self.recoil_control_enabled.setChecked(True)
                self.edited_config['agent'] = agent_name
                save_config(self.edited_config)
                index = self.select_agent_manual.findText(agent_name)
                if index != -1:
                    self.select_agent_manual.setCurrentIndex(index)
            else:
                logger.warning(f"ConfigEditor: Failed to activate Recoil Control for agent: {agent_name}")

    def get_all_agents(self):
        profiles_path = PROFILES_PATH
        try:
            profiles_dict = load_profiles(profiles_path)
            agents = sorted(profiles_dict.keys())
            return agents
        except Exception as e:
            logger.error(f"ConfigEditor: Error loading agents for manual selection: {e}")
            return []

    def select_agent_manually(self, index):
        if index == 0:
            return
        agent_name = self.select_agent_manual.currentText().upper()
        logger.info(f"ConfigEditor: Agent selected manually: {agent_name}")
        self.comm.agent_selected_manually.emit(agent_name)

    def on_agent_selected_manually(self, agent_name):
        if self.recoil_control:
            self.recoil_control.set_profile(agent_name)
            if self.recoil_control.profile_data:
                logger.info(f"ConfigEditor: Recoil Control set to agent: {agent_name}")
                self.edited_config['agent'] = agent_name
                save_config(self.edited_config)
            else:
                logger.warning(f"ConfigEditor: Failed to set Recoil Control for agent: {agent_name}")

    def edit_recoil_strength(self):
        agent_name = self.edited_config.get('agent', 'THERMITE').upper()
        if not agent_name:
            QMessageBox.warning(self, "Warning", "No agent selected to edit recoil strength.")
            return
        profile = self.recoil_control.profiles.get(agent_name, None) if self.recoil_control else None
        if not profile:
            QMessageBox.warning(self, "Warning", f"No profile found for agent '{agent_name}'.")
            return
        current_y = profile.get("y", 0)
        dialog = RecoilStrengthEditor(current_y, self)
        if dialog.exec_():
            new_y = dialog.new_y
            self.update_profile_y(agent_name, new_y)
            if self.recoil_control:
                self.recoil_control.set_profile(agent_name)
            logger.info(f"ConfigEditor: Recoil Strength for '{agent_name}' updated to {new_y}.")
            self.comm.recoil_strength_updated.emit(new_y)

    def update_profile_y(self, agent_name, new_y):
        profiles_path = PROFILES_PATH
        try:
            with open(profiles_path, 'r') as profiles_file:
                profiles = json.load(profiles_file)
            for profile in profiles:
                if profile["profile"].upper() == agent_name:
                    profile["y"] = new_y
                    break
            with open(profiles_path, 'w') as profiles_file:
                json.dump(profiles, profiles_file, indent=4)
            logger.info(f"ConfigEditor: Updated 'y' value for '{agent_name}' in profiles.json.")
        except Exception as e:
            logger.error(f"ConfigEditor: Error updating 'y' value for '{agent_name}': {e}")
            QMessageBox.critical(self, "Error", f"Failed to update recoil strength for '{agent_name}':\n{e}")

    def change_aspect_ratio(self, value):
        self.edited_config["selected_aspect_ratio"] = value
        logger.info(f"ConfigEditor: Selected Aspect Ratio set to: {value}")

    def change_model_path(self, text):
        self.edited_config["model_path"] = text
        if self.aimbot:
            self.aimbot.update_config(self.edited_config)
        logger.info(f"ConfigEditor: Model Path set to: {text}")
        model_path = os.path.join(script_dir, text)
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Warning", f"Model file not found at {model_path}. Please ensure it exists in the script's directory.")

    def change_move_mouse_steps(self, value):
        if "move_mouse" not in self.edited_config:
            self.edited_config["move_mouse"] = {"steps": 40, "delay": 0.0009}
        self.edited_config["move_mouse"]["steps"] = value
        if self.aimbot:
            self.aimbot.update_config(self.edited_config)
        logger.info(f"ConfigEditor: Move Mouse Steps set to: {value}")

    def change_move_mouse_delay(self, value):
        if "move_mouse" not in self.edited_config:
            self.edited_config["move_mouse"] = {"steps": 40, "delay": 0.0009}
        self.edited_config["move_mouse"]["delay"] = value
        if self.aimbot:
            self.aimbot.update_config(self.edited_config)
        logger.info(f"ConfigEditor: Move Mouse Delay set to: {value}")

    def change_aimbot_key(self, key):
        self.edited_config["keys"]["activate"] = key.lower()
        if self.aimbot:
            self.aimbot.update_config(self.edited_config)
        logger.info(f"ConfigEditor: Aimbot Activation Key set to: {key}")

    def browse_model_path(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Model File", script_dir, "Model Files (*.pt *.pth)")
        if file_name:
            relative_path = os.path.relpath(file_name, script_dir)
            self.model_path.setText(relative_path)
            logger.info(f"ConfigEditor: Selected Model Path: {relative_path}")

    def save_config(self):
        save_config(self.edited_config)
        self.config = copy.deepcopy(self.edited_config)
        if self.aimbot:
            self.aimbot.update_config(self.config)
        if self.recoil_control:
            self.recoil_control.update_config(self.config)
        QMessageBox.information(self, "Success", "Configuration saved successfully.")
        logger.info("ConfigEditor: Configuration saved via UI.")

    def cancel_changes(self):
        self.edited_config = copy.deepcopy(self.config)
        QMessageBox.information(self, "Canceled", "All unsaved changes have been discarded.")
        logger.info("ConfigEditor: Unsaved changes discarded.")
        self.close()

    def on_config_updated(self):
        pass

    def on_aimbot_toggled(self, enabled):
        logger.info(f"ConfigEditor: Aimbot toggled: {enabled}")

    def on_trigger_bot_toggled(self, enabled):
        logger.info(f"ConfigEditor: Trigger Bot toggled via checkbox: {enabled}")

    def on_trigger_bot_active_toggled(self, active):
        logger.info(f"ConfigEditor: Trigger Bot active state toggled via Caps Lock: {active}")
        status = "Active" if active else "Inactive"
        self.trigger_bot_info_label.setText(f"(Trigger Bot is {status}. Press Caps Lock to toggle.)")
        if self.aimbot:
            self.aimbot.update_config(self.edited_config)

    def on_recoil_control_toggled(self, enabled):
        logger.info(f"ConfigEditor: Recoil Control toggled: {enabled}")

    def on_recoil_strength_updated(self, new_y):
        logger.info(f"ConfigEditor: Recoil Strength updated to: {new_y}")
        QTimer.singleShot(100, self.toggle_recoil_control_off_and_on)

    def toggle_recoil_control_off_and_on(self):
        self.recoil_control_enabled.setChecked(False)
        QTimer.singleShot(100, lambda: self.recoil_control_enabled.setChecked(True))

    def closeEvent(self, event):
        self.stop_aimbot()
        self.stop_recoil_control()
        logger.info("ConfigEditor: Application is closing. Background threads stopped and hotkeys unbound.")
        event.accept()

def main():
    url1 = "https://www.google.com/search?q=site%3Agamerfun.club+gamerfun+ai+aimbot"
    webbrowser.open_new_tab(url1)

    url2 = "https://www.gamerfun.club/ai-aimbot-triggerbot-shooter-games"
    webbrowser.open_new_tab(url2)
    app = QApplication(sys.argv)
    style_sheet = """
    QWidget {
        background-color: #2b2b2b;
        color: #FFFFFF;
        font-size: 14px;
    }

    QLabel {
        color: #32CD32;
    }

    QGroupBox {
        border: 1px solid #32CD32;
        margin-top: 10px;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px 0 3px;
    }

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background-color: #3c3c3c;
        color: #FFFFFF;
        border: 1px solid #32CD32;
        padding: 2px;
        border-radius: 5px;
    }

    QCheckBox {
        color: #FFFFFF;
    }

    QPushButton {
        background-color: #32CD32;
        color: #FFFFFF;
        border: none;
        padding: 8px;
        border-radius: 5px;
    }

    QPushButton:hover {
        background-color: #28a428;
    }

    QScrollBar:vertical, QScrollBar:horizontal {
        background-color: #3c3c3c;
        width: 15px;
        margin: 15px 3px 15px 3px;
        border: 1px solid #32CD32;
    }

    QScrollBar::handle {
        background-color: #32CD32;
        min-height: 20px;
        border-radius: 7px;
    }

    QComboBox QAbstractItemView {
        background-color: #3c3c3c;
        selection-background-color: #32CD32;
        selection-color: #FFFFFF;
        border: 1px solid #32CD32;
    }

    QToolTip {
        background-color: #3c3c3c;
        color: #FFFFFF;
        border: 1px solid #32CD32;
    }
    """

    app.setStyleSheet(style_sheet)
    editor = ConfigEditor()
    editor.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
