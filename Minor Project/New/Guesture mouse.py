import cv2
import numpy as np
import mediapipe as mp
import pyautogui

# --- CONFIGURATION ---
wCam, hCam = 640, 480      # Webcam resolution
frameR = 100               # Frame Reduction (Padding). Hand must be inside this box.
smoothening = 10           # Smoothening factor (Higher = smoother but slightly slower)

# --- SETUP ---
cap = cv2.VideoCapture(0)
cap.set(3, wCam) # Width
cap.set(4, hCam) # Height

pTime = 0
plocX, plocY = 0, 0        # Previous location (for smoothing)
clocX, clocY = 0, 0        # Current location (for smoothing)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1) # Detect only 1 hand
mp_draw = mp.solutions.drawing_utils

# Get actual screen size of your laptop
wScr, hScr = pyautogui.size()

print("Virtual Mouse Starting... Press 'q' to quit.")

while True:
    # 1. Find Hand Landmarks
    success, img = cap.read()
    if not success:
        break
        
    img = cv2.flip(img, 1) # Mirror the image so it feels natural
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # Draw the "Active Zone" Box
    # You only need to move your hand inside this rectangle to cover the whole screen
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                  (255, 0, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            lmList = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            # We need at least the thumb and index finger tips
            if len(lmList) != 0:
                x1, y1 = lmList[8][1:]  # Index Finger Tip (ID 8)
                x2, y2 = lmList[4][1:]  # Thumb Tip (ID 4)
                
                # 2. Convert Coordinates (Interpolation)
                # Converts the webcam coordinates to your screen's resolution
                # np.interp(value, [min_input, max_input], [min_output, max_output])
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                
                # 3. Smoothen Values (To remove jitter)
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                
                # 4. Move Mouse
                # We use clocX/Y (Smoothed values) instead of raw x3/y3
                # (try/except block handles edge-of-screen errors)
                try:
                    pyautogui.moveTo(clocX, clocY)
                except:
                    pass
                
                # Visual feedback for the pointer
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY
                
                # 5. Clicking Mode (Check distance between Index and Thumb)
                length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5 # Calculate distance
                
                # If fingers are close enough (Distance < 40 pixels), trigger click
                if length < 40:
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED) # Turn Green
                    pyautogui.click()
    
    # Display Frame
    cv2.imshow("AI Virtual Mouse", img)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()