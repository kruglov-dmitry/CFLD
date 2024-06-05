import cv2
import dlib
import numpy as np

# Load the images
real_image = cv2.imread('images_to_play_with/fd_sm.jpg')
generated_image = cv2.imread('images_to_play_with/target_pose_3_with_face.jpeg')

# Initialize face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Detect faces
real_faces = face_detector(real_image, 1)
generated_faces = face_detector(generated_image, 1)

# Get the first face
real_face = real_faces[0]
generated_face = generated_faces[0]

# Get facial landmarks
real_landmarks = predictor(real_image, real_face)
generated_landmarks = predictor(generated_image, generated_face)

# Convert landmarks to numpy arrays
def landmarks_to_np(landmarks):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords

real_landmarks_np = landmarks_to_np(real_landmarks)
generated_landmarks_np = landmarks_to_np(generated_landmarks)

# Align face
def align_face(image, src_landmarks, dst_landmarks):
    h, status = cv2.findHomography(src_landmarks, dst_landmarks)
    aligned_face = cv2.warpPerspective(image, h, (image.shape[1], image.shape[0]))
    return aligned_face

aligned_face = align_face(real_image, real_landmarks_np, generated_landmarks_np)

# Blend face
(x, y, w, h) = (generated_face.left(), generated_face.top(), generated_face.width(), generated_face.height())
face_region = generated_image[y:y+h, x:x+w]
aligned_face_cropped = aligned_face[y:y+h, x:x+w]

mask = 255 * np.ones(aligned_face_cropped.shape, aligned_face_cropped.dtype)
center = (x + w//2, y + h//2)

output = cv2.seamlessClone(aligned_face_cropped, generated_image, mask, center, cv2.NORMAL_CLONE)

# Save or display the final result
cv2.imwrite('final_image.jpg', output)
cv2.imshow('Final Image', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
