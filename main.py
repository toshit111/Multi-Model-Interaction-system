import cv2
import mediapipe as mp
import pyautogui
import time
import math
from pynput.mouse import Button, Controller

# Disable PyAutoGUI's fail-safe feature
pyautogui.FAILSAFE = False  # Disables the fail-safe, which stops code if the mouse moves to the corner of the screen

# Initialize camera
cam = cv2.VideoCapture(0)

# Initialize Mediapipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Initialize Mediapipe Hands solution
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Initialize mouse controller for finger control
mouse = Controller()

# Get screen dimensions
screen_w, screen_h = pyautogui.size()
screen_width, screen_height = pyautogui.size()

# Move pointer to the center of the screen at the start (eye control)
pyautogui.moveTo(screen_w // 2, screen_h // 2)
# Initialize previous cursor position for smoothness
prev_screen_x, prev_screen_y = screen_w // 2, screen_h // 2


#-----------------------------------------------------------------------------------------------------------------
# Time thresholds and blink detection variables (eye control)
LONG_CLOSE_THRESHOLD = 0.7  # Time in seconds for long close detection
BLINK_RESET_TIME = 1.0  # Time to reset blink detection
BLINK_THRESHOLD = 0.004  # Threshold for eye blink detection

# Adjust cursor sensitivity for hand tracking
HAND_CURSOR_SENSITIVITY = 1.5  # Increase sensitivity (higher values = faster movement)

#------------------------------------------------------------------------------------------------------------------

# Eye blink detection state and timers
left_eye_blink_time = None
right_eye_blink_time = None
both_eyes_close_time = None
left_eye_blink_detected = False
right_eye_blink_detected = False

#--------------------------------------------------------------------------------------------------------------------

# Cursor sensitivity for head movement (eye control)
CURSOR_SENSITIVITY = 3.5  # Adjust to make smaller head movements more effective
SMOOTHING_FACTOR = 0.5  # A factor between 0 and 1 to control smoothness (lower = smoother)

#----------------------------------------------------------------------------------------------------------------------

# Variable to store the last mouse position (finger control)
last_mouse_position = None


# Function to detect if the eye is blinking
def is_eye_blinking(eye_landmarks):
    upper_eye = eye_landmarks[0]
    lower_eye = eye_landmarks[1]
    return (upper_eye.y - lower_eye.y) < BLINK_THRESHOLD

def move_cursor_smoothly(nose_landmark):
    global prev_screen_x, prev_screen_y

    # Calculate the target position for the cursor
    target_screen_x = int((nose_landmark.x - 0.5) * screen_w * CURSOR_SENSITIVITY)
    target_screen_y = int((nose_landmark.y - 0.5) * screen_h * CURSOR_SENSITIVITY)

    # Apply smoothing by blending the previous position and the target position
    screen_x = int(prev_screen_x + SMOOTHING_FACTOR * (target_screen_x - prev_screen_x))
    screen_y = int(prev_screen_y + SMOOTHING_FACTOR * (target_screen_y - prev_screen_y))

    # Constrain the cursor within screen bounds
    screen_x = min(max(screen_x, 0), screen_w - 1)
    screen_y = min(max(screen_y, 0), screen_h - 1)

    # Move the cursor
    pyautogui.moveTo(screen_x, screen_y)

    # Update the previous position for the next frame
    prev_screen_x, prev_screen_y = screen_x, screen_y



# Function to move cursor based on head movement (eye control)
def move_cursor_based_on_head(nose_landmark):
    # Apply sensitivity to the cursor movement
    screen_x = int((nose_landmark.x - 0.5) * screen_w * CURSOR_SENSITIVITY)
    screen_y = int((nose_landmark.y - 0.5) * screen_h * CURSOR_SENSITIVITY)

    # Constrain the cursor within screen bounds
    screen_x = min(max(screen_x, 0), screen_w - 1)
    screen_y = min(max(screen_y, 0), screen_h - 1)

    pyautogui.moveTo(screen_x, screen_y)


# Function to find the tip of the index finger (finger control)
def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None



# Move the mouse according to the index finger's position (finger control)
def move_mouse(index_finger_tip):
    global last_mouse_position
    if index_finger_tip is not None:
        # Scale the hand landmark coordinates to the screen dimensions
        x = int(index_finger_tip.x * screen_width * HAND_CURSOR_SENSITIVITY)
        y = int(index_finger_tip.y * screen_height * HAND_CURSOR_SENSITIVITY)

        # Constrain the cursor within screen bounds
        x = min(max(x, 0), screen_width - 1)
        y = min(max(y, 0), screen_height - 1)

        # Move the mouse cursor
        pyautogui.moveTo(x, y)

        # Save the current mouse position for gesture detection
        last_mouse_position = (x, y)



# Calculate Euclidean distance between two landmarks (finger control)
def get_distance(p1, p2):
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


# Calculate the angle between three landmarks (finger control)
def get_angle(p1, p2, p3):
    a = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
    b = math.sqrt((p2.x - p3.x) ** 2 + (p2.y - p3.y) ** 2)
    c = math.sqrt((p3.x - p1.x) ** 2 + (p3.y - p1.y) ** 2)
    if a == 0 or b == 0:
        return 0
    angle = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    return math.degrees(angle)


# Check if the thumb is closed (near the index finger) for finger control
def is_thumb_closed(landmark_list):
    thumb_tip = landmark_list[mpHands.HandLandmark.THUMB_TIP]
    index_finger_base = landmark_list[mpHands.HandLandmark.INDEX_FINGER_MCP]
    distance = get_distance(thumb_tip, index_finger_base)
    return distance < 0.1  # Threshold for thumb being "closed"


# Check if a left click gesture is detected (finger control)
def is_left_click(landmark_list):
    return (
            get_angle(landmark_list[mpHands.HandLandmark.INDEX_FINGER_MCP],
                      landmark_list[mpHands.HandLandmark.INDEX_FINGER_PIP],
                      landmark_list[mpHands.HandLandmark.INDEX_FINGER_TIP]) < 50 and
            get_angle(landmark_list[mpHands.HandLandmark.MIDDLE_FINGER_MCP],
                      landmark_list[mpHands.HandLandmark.MIDDLE_FINGER_PIP],
                      landmark_list[mpHands.HandLandmark.MIDDLE_FINGER_TIP]) > 90
    )


# Check if a right click gesture is detected (finger control)
def is_right_click(landmark_list):
    return (
            get_angle(landmark_list[mpHands.HandLandmark.MIDDLE_FINGER_MCP],
                      landmark_list[mpHands.HandLandmark.MIDDLE_FINGER_PIP],
                      landmark_list[mpHands.HandLandmark.MIDDLE_FINGER_TIP]) < 50 and
            get_angle(landmark_list[mpHands.HandLandmark.INDEX_FINGER_MCP],
                      landmark_list[mpHands.HandLandmark.INDEX_FINGER_PIP],
                      landmark_list[mpHands.HandLandmark.INDEX_FINGER_TIP]) > 90
    )


# Detect and perform gestures (finger control)
def detect_gesture(frame, landmark_list, processed):
    global last_mouse_position

    if len(landmark_list) >= 21:
        index_finger_tip = find_finger_tip(processed)

        # If thumb is closed, allow cursor movement
        if is_thumb_closed(landmark_list):
            move_mouse(index_finger_tip)
        else:
            # When thumb is open, stop movement and allow clicks
            if is_left_click(landmark_list):
                if last_mouse_position:
                    pyautogui.moveTo(last_mouse_position[0], last_mouse_position[1])  # Keep the cursor still
                mouse.press(Button.left)
                mouse.release(Button.left)
                cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif is_right_click(landmark_list):
                if last_mouse_position:
                    pyautogui.moveTo(last_mouse_position[0], last_mouse_position[1])  # Keep the cursor still
                mouse.press(Button.right)
                mouse.release(Button.right)
                cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


# Function to switch control between eyes and fingers based on hand detection
def switch_control(hand_detected, eye_control_func, finger_control_func):
    if hand_detected:
        # If hand is detected, switch to finger control
        finger_control_func()
    else:
        # If no hand is detected, switch to eye control
        eye_control_func()


# Main loop
while True:
    ret, frame = cam.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Failed to capture frame, retrying...")
        continue

    # Flip the frame for a mirrored view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the face landmarks for eye detection
    output_face = face_mesh.process(rgb_frame)
    landmark_points = output_face.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    # Process the hand landmarks for finger detection
    output_hand = hands.process(rgb_frame)
    hand_detected = output_hand.multi_hand_landmarks is not None


    # Eye control function (head movement and blink detection)
    def eye_control():
        global left_eye_blink_detected, right_eye_blink_detected, left_eye_blink_time, right_eye_blink_time
        if landmark_points:  # Process eye landmarks if detected
            landmarks = landmark_points[0].landmark

            # Nose landmark for cursor control (landmark 1)
            nose_landmark = landmarks[1]
            move_cursor_smoothly(nose_landmark)  # Use smooth movement for cursor

            # Left eye (landmarks 145 and 159)
            left_eye = [landmarks[145], landmarks[159]]
            left_eye_blink = is_eye_blinking(left_eye)

            # Right eye (landmarks 374 and 386)
            right_eye = [landmarks[374], landmarks[386]]
            right_eye_blink = is_eye_blinking(right_eye)

            # Draw landmarks for left eye (with yellow color)
            for landmark in left_eye:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Yellow circles for left eye

            # Draw landmarks for right eye (with cyan color)
            for landmark in right_eye:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)  # Cyan circles for right eye

            # Blink detection and actions
            if left_eye_blink and not left_eye_blink_detected:
                left_eye_blink_detected = True
                left_eye_blink_time = time.time()
            elif not left_eye_blink and left_eye_blink_detected:
                if time.time() - left_eye_blink_time < LONG_CLOSE_THRESHOLD:
                    pyautogui.click()
                left_eye_blink_detected = False

            if right_eye_blink and not right_eye_blink_detected:
                right_eye_blink_detected = True
                right_eye_blink_time = time.time()
            elif not right_eye_blink and right_eye_blink_detected:
                if time.time() - right_eye_blink_time < LONG_CLOSE_THRESHOLD:
                    pyautogui.click(button='right')
                right_eye_blink_detected = False


    # Finger control function (hand gesture-based control)
    def finger_control():
        if output_hand.multi_hand_landmarks:  # Process hand landmarks if detected
            hand_landmarks = output_hand.multi_hand_landmarks[0]
            detect_gesture(frame, hand_landmarks.landmark, output_hand)


    # Call the switch function to determine control mode
    switch_control(hand_detected, eye_control, finger_control)

    # Draw the hand landmarks on the frame (finger control)
    if output_hand.multi_hand_landmarks:
        for hand_landmarks in output_hand.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Eye and Finger Control", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cam.release()
cv2.destroyAllWindows()
