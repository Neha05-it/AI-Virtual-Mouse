import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import comtypes
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --- CONFIGURATION ---
wCam, hCam = 640, 480       # Webcam Resolution
frameR = 100                # Frame Reduction
smoothening = 7             # Mouse Smoothening

# --- SETUP WINDOWS COM (Crucial Fix) ---
try:
    comtypes.CoInitialize()
except:
    pass # Already initialized

# --- VOLUME SETUP (Pycaw) ---
volume = None
volBar = 400
volPer = 0
minVol = 0
maxVol = 0

try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange() 
    minVol = volRange[0]
    maxVol = volRange[1]
    print("✅ Audio System Connected Successfully.")
except Exception as e:
    print(f"⚠️ Warning: Audio control unavailable ({e}). Mouse will work, but Volume Hand won't.")
    volume = None

# --- CV SETUP ---
# Try index 1 if index 0 fails
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2) # Enable 2 Hands
mp_draw = mp.solutions.drawing_utils

wScr, hScr = pyautogui.size()
plocX, plocY = 0, 0
clocX, clocY = 0, 0

last_click_time = 0

print("Virtual Mouse ULTIMATE (Dual Hand) Starting... Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.flip(img, 1) # Mirror view
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # Draw Reference Box for Mouse
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    if results.multi_hand_landmarks:
        # Iterate through all detected hands
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            
            # Determine which hand it is (Left or Right)
            # Note: In mirror mode, 'Left' label usually means User's Right Hand
            hand_label = handedness.classification[0].label 
            
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            lmList = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                # Get Finger Tips
                x1, y1 = lmList[8][1:]   # Index
                x2, y2 = lmList[4][1:]   # Thumb

                # =====================================================
                #  RIGHT HAND (USER'S RIGHT) -> MOUSE CONTROL
                # =====================================================
                if hand_label == "Left": # This is usually the User's Right Hand in Selfie Mode
                    
                    # 1. Move Mouse (Index Finger)
                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                    
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    
                    try:
                        pyautogui.moveTo(clocX, clocY)
                    except:
                        pass
                    plocX, plocY = clocX, clocY

                    # 2. Clicking (Thumb + Index)
                    length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                    if length < 40:
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                        if time.time() - last_click_time > 0.5:
                            pyautogui.click()
                            last_click_time = time.time()
                            
                    # 3. Right Click (Middle Finger + Thumb)
                    x3_mid, y3_mid = lmList[12][1:]
                    length_right = ((x2 - x3_mid)**2 + (y2 - y3_mid)**2)**0.5
                    if length_right < 40:
                        cv2.circle(img, (x3_mid, y3_mid), 15, (0, 0, 255), cv2.FILLED)
                        if time.time() - last_click_time > 0.5:
                            pyautogui.rightClick()
                            last_click_time = time.time()

                # =====================================================
                #  LEFT HAND (USER'S LEFT) -> VOLUME CONTROL
                # =====================================================
                if hand_label == "Right" and volume is not None: # Only run if volume loaded
                    
                    # Calculate distance between Thumb (4) and Index (8)
                    length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                    
                    # Visual feedback line
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

                    # Map Distance (Range 30 to 200) to Volume Range
                    vol = np.interp(length, [30, 200], [minVol, maxVol])
                    volBar = np.interp(length, [30, 200], [400, 150])
                    volPer = np.interp(length, [30, 200], [0, 100])
                    
                    # Set System Volume
                    volume.SetMasterVolumeLevel(vol, None)
                    
                    # Visual Volume Bar UI
                    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 
                                1, (0, 250, 0), 3)

    cv2.imshow("AI Virtual Mouse Ultimate", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()