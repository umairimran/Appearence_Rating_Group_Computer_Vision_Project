import cv2
import mediapipe as mp
import numpy as np

# Load cropped image
cropped_img = cv2.imread("cropped_people/person_2.jpg")

# Initialize MediaPipe Selfie Segmentation
mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

# Convert to RGB
image_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

# Run segmentation
results = segmentor.process(image_rgb)

# Get the mask (0.0 to 1.0)
mask = results.segmentation_mask

# Resize mask if needed
mask = cv2.resize(mask, (cropped_img.shape[1], cropped_img.shape[0]))

# Create 3-channel alpha mask
mask_3ch = np.stack((mask,) * 3, axis=-1)

# Background (white)
background = np.ones_like(cropped_img, dtype=np.uint8) * 255

# Alpha blending
final_image = (mask_3ch * cropped_img + (1 - mask_3ch) * background).astype(np.uint8)

# Save the result
cv2.imwrite("segmented_person_soft.jpg", final_image)
