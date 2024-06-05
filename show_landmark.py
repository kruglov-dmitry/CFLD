import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe drawing utils
mp_drawing = mp.solutions.drawing_utils

# Read an image
#image = cv2.imread('./images_to_play_with/fd_sm.jpg')
image = cv2.imread('./images_to_play_with/target_pose_3.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and get pose landmarks
result = pose.process(image_rgb)

# Draw landmarks on the image
if result.pose_landmarks:
    mp_drawing.draw_landmarks(
        image,
        result.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Landmarks
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   # Connections
    )

# Convert the image back to BGR for OpenCV
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Display the image
cv2.imshow('Pose Estimation', image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
