import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_gesture = None
gesture_time = 0

def count_fingers(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x)

    # Other fingers
    for i in range(1, 5):
        fingers.append(
            hand_landmarks.landmark[tips[i]].y <
            hand_landmarks.landmark[tips[i]-2].y
        )
    return fingers.count(True)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            finger_count = count_fingers(handLms)
            current_time = time.time()

            if current_time - gesture_time > 1:
                if finger_count == 2 and prev_gesture != "next":
                    pyautogui.press("right")
                    prev_gesture = "next"
                    gesture_time = current_time

                elif finger_count == 1 and prev_gesture != "prev":
                    pyautogui.press("left")
                    prev_gesture = "prev"
                    gesture_time = current_time

    cv2.imshow("Hand Controlled Presentation", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
