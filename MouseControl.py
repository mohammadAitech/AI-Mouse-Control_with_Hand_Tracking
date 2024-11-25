import mediapipe as mp
import cv2 as cv
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)

    if not ret:
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            index_finger_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            screen_width, screen_height= pyautogui.size()
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            pyautogui.moveTo(x,y)

            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

    cv.imshow("hand tracking", frame)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()