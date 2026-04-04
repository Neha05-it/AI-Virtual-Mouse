import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

# --- CONFIGURATION ---
wCam, hCam = 640, 480       # Webcam resolution
frameR = 100              # Frame Reduction (Padding)
smoothening = 4           # Smoothening factor

# --- SETUP ---
cap = cv2.VideoCapture(0)   # Try index 1 if 0 fails
cap.set(3, wCam)
cap.set(4, hCam)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

wScr, hScr = pyautogui.size()
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# State variables for clicks and gestures
last_click_time = 0
is_alt_tab_active = False
pinch_start_x = 0           # To track swipe movement

print("Virtual Mouse 3.0 (Alt-Tab Edition) Starting... Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break
        
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # Draw the "Active Zone" Box
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            lmList = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            if len(lmList) != 0:
                # --- GET FINGER TIPS ---
                x1, y1 = lmList[8][1:]   # Index (Move)
                x2, y2 = lmList[4][1:]   # Thumb (Trigger)
                x3, y3 = lmList[12][1:]  # Middle (Right Click)
                x4, y4 = lmList[16][1:]  # Ring (Double Click)
                x5, y5 = lmList[20][1:]  # Pinky (Alt-Tab)

                # --- CHECK ALT-TAB GESTURE (Pinky + Thumb) ---
                dis_alt = ((x2 - x5)**2 + (y2 - y5)**2)**0.5
                
                if dis_alt < 40: # Gesture Active
                    if not is_alt_tab_active:
                        # Activate Alt-Tab Mode
                        pyautogui.keyDown('alt')
                        pyautogui.press('tab')
                        is_alt_tab_active = True
                        pinch_start_x = x1 # Record starting X position
                    
                    # Visual Indicator (Blue Dot on Pinky)
                    cv2.circle(img, (x5, y5), 15, (255, 0, 0), cv2.FILLED)
                    
                    # --- SWIPE LOGIC ---
                    # Move Right (> 50px from start)
                    if x1 > pinch_start_x + 50:
                        pyautogui.press('tab')
                        pinch_start_x = x1 # Reset start to allow continuous scrolling
                        
                    # Move Left (< 50px from start)
                    elif x1 < pinch_start_x - 50:
                        pyautogui.hotkey('shift', 'tab')
                        pinch_start_x = x1

                else: # Gesture NOT Active
                    if is_alt_tab_active:
                        # Release Alt-Tab Mode
                        pyautogui.keyUp('alt')
                        is_alt_tab_active = False
                    
                    # --- NORMAL MOUSE MODE (Only when not Alt-Tabbing) ---
                    
                    # 1. Move Mouse (Index Finger)
                    x3_map = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3_map = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                    
                    clocX = plocX + (x3_map - plocX) / smoothening
                    clocY = plocY + (y3_map - plocY) / smoothening
                    
                    try:
                        pyautogui.moveTo(clocX, clocY)
                    except:
                        pass
                    plocX, plocY = clocX, clocY
                    
                    # 2. Check Click Distances
                    dis_left = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                    dis_right = ((x2 - x3)**2 + (y2 - y3)**2)**0.5
                    dis_double = ((x2 - x4)**2 + (y2 - y4)**2)**0.5
                    
                    current_time = time.time()
                    
                    # Left Click (Index + Thumb)
                    if dis_left < 40:
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                        if current_time - last_click_time > 0.5:
                            pyautogui.click()
                            last_click_time = current_time

                    # Right Click (Middle + Thumb)
                    elif dis_right < 40:
                        cv2.circle(img, (x3, y3), 15, (0, 0, 255), cv2.FILLED)
                        if current_time - last_click_time > 0.5:
                            pyautogui.rightClick()
                            last_click_time = current_time
                            
                    # Double Click (Ring + Thumb)
                    elif dis_double < 40:
                        cv2.circle(img, (x4, y4), 15, (255, 255, 0), cv2.FILLED)
                        if current_time - last_click_time > 1.0:
                            pyautogui.doubleClick()
                            last_click_time = current_time

    cv2.imshow("AI Virtual Mouse 3.0", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()