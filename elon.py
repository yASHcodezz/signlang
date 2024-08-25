import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function to detect hand gestures
def detect_hand_gestures(hand_landmarks):
    # Example logic for recognizing a gesture
    # Calculate distances between specific landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Example gesture recognition logic
    if thumb_tip.y < index_tip.y:  # Simple logic for a gesture (thumbs up)
        return "Thumbs Up"
    else:
        return "Unknown Gesture"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Convert the frame to RGB as MediaPipe uses RGB images
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hand landmarks
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detect the gesture
            gesture = detect_hand_gestures(hand_landmarks)
            cv2.putText(frame, gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow("Hand Gesture Recognition", frame)
    
    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
