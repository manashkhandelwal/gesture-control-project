import cv2
import mediapipe as mp
import pydirectinput as pyautogui
import time

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)

tipIds = [4, 8, 12, 16, 20]

# ---- CONTROL VARIABLES ----
active_key = None

# Stability buffer
gesture_history = []
history_length = 5

# ---- FUNCTIONS ----
def get_finger_state(lmList, label):
    fingers = []

    # Thumb
    if label == "Right":
        fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)
    else:
        fingers.append(1 if lmList[4][1] < lmList[3][1] else 0)

    # Other fingers
    for i in range(1, 5):
        fingers.append(1 if lmList[tipIds[i]][2] < lmList[tipIds[i] - 2][2] else 0)

    return fingers


def update_key(new_key):
    global active_key

    if new_key == active_key:
        return

    # Release old key
    if active_key is not None:
        pyautogui.keyUp(active_key)

    # Press new key
    if new_key is not None:
        pyautogui.keyDown(new_key)

    active_key = new_key


# ---- MAIN LOOP ----
try:
    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2
    ) as hands:

        print("Starting in 5 seconds... switch to your game window!")
        time.sleep(5)

        while True:
            ret, image = video.read()
            if not ret:
                break

            image = cv2.flip(image, 1)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            hand_states = {}

            if results.multi_hand_landmarks and results.multi_handedness:

                for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness):

                    label = handedness.classification[0].label
                    h, w, _ = image.shape

                    lmList = []
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([idx, cx, cy])

                    fingers = get_finger_state(lmList, label)
                    total = sum(fingers)

                    if total >= 4:
                        state = "open"
                    elif total <= 1:
                        state = "closed"
                    else:
                        state = "neutral"

                    hand_states[label] = state

                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    cv2.putText(
                        image,
                        f"{label}: {state}",
                        (10, 40 if label == "Left" else 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2
                    )

            # ---- DECIDE ACTION ----
            new_key = None

            if "Left" in hand_states and "Right" in hand_states:
                left = hand_states["Left"]
                right = hand_states["Right"]

                if left == "open" and right == "open":
                    new_key = "w"
                elif left == "closed" and right == "closed":
                    new_key = "s"
                elif right == "closed" and left == "open":
                    new_key = "d"
                elif left == "closed" and right == "open":
                    new_key = "a"

            # ---- STABILITY FILTER ----
            gesture_history.append(new_key)

            if len(gesture_history) > history_length:
                gesture_history.pop(0)

            if gesture_history.count(gesture_history[0]) == len(gesture_history):
                stable_key = gesture_history[0]
            else:
                stable_key = None

            # ---- APPLY KEY ----
            update_key(stable_key)

            # Debug text
            cv2.putText(
                image,
                f"Action: {stable_key}",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            cv2.imshow("Gesture WASD Control", image)

            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:
    if active_key is not None:
        pyautogui.keyUp(active_key)

    video.release()
    cv2.destroyAllWindows()