# Importing Libraries 
import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np
import pyautogui
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize Audio
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

Draw = mp.solutions.drawing_utils

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

mode = "Brightness"  # default mode

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Process = hands.process(frameRGB)

    landmarkList = []
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                height, width, _ = frame.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])

            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    if landmarkList != []:
        # Finger states (tip ids: thumb=4, index=8, middle=12, ring=16, pinky=20)
        fingers = []
        tipIds = [4, 8, 12, 16, 20]
        for i in range(1, 5):
            if landmarkList[tipIds[i]][2] < landmarkList[tipIds[i]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

        # Mode switching
        if totalFingers == 1:
            mode = "Brightness"
        elif totalFingers == 2:
            mode = "Volume"
        elif totalFingers == 3:
            mode = "Mouse"

        cv2.putText(frame, f"Mode: {mode}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Mode actions
        if mode == "Brightness":
            x1, y1 = landmarkList[4][1], landmarkList[4][2]
            x2, y2 = landmarkList[8][1], landmarkList[8][2]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            L = hypot(x2 - x1, y2 - y1)
            b_level = np.interp(L, [15, 220], [0, 100])
            sbc.set_brightness(int(b_level))

            cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(frame, (50, int(400 - b_level*2.5)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f'Brightness: {int(b_level)}%', (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif mode == "Volume":
            x2, y2 = landmarkList[8][1], landmarkList[8][2]
            x3, y3 = landmarkList[12][1], landmarkList[12][2]
            cv2.line(frame, (x2, y2), (x3, y3), (255, 0, 0), 3)
            L2 = hypot(x3 - x2, y3 - y2)
            vol_level = np.interp(L2, [15, 220], [volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]])
            volume.SetMasterVolumeLevel(vol_level, None)

            vol_percent = np.interp(vol_level, [volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]], [0, 100])
            cv2.rectangle(frame, (550, 150), (585, 400), (255, 0, 0), 3)
            cv2.rectangle(frame, (550, int(400 - vol_percent*2.5)), (585, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(frame, f'Volume: {int(vol_percent)}%', (530, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        elif mode == "Mouse":
            x_index, y_index = landmarkList[8][1], landmarkList[8][2]
            screen_w, screen_h = pyautogui.size()
            mouse_x = np.interp(x_index, [0, frame.shape[1]], [0, screen_w])
            mouse_y = np.interp(y_index, [0, frame.shape[0]], [0, screen_h])
            pyautogui.moveTo(mouse_x, mouse_y)

            # Click if thumb and index are close
            x_thumb, y_thumb = landmarkList[4][1], landmarkList[4][2]
            distance = hypot(x_index - x_thumb, y_index - y_thumb)
            if distance < 30:
                pyautogui.click()

    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
