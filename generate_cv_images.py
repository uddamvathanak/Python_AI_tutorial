import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

# Ensure the directory exists
os.makedirs('static/images/cv', exist_ok=True)

# 1. Create a sample image for basic processing examples
def create_sample_image():
    # Create a simple image with shapes
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Add shapes of different colors
    # Blue rectangle
    cv2.rectangle(img, (100, 50), (250, 200), (255, 0, 0), -1)
    
    # Green circle
    cv2.circle(img, (400, 120), 80, (0, 255, 0), -1)
    
    # Red triangle
    pts = np.array([[300, 300], [200, 380], [400, 380]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 255))
    
    # Yellow star in the bottom right
    center = (480, 320)
    radius = 60
    num_points = 5
    points = []
    for i in range(num_points * 2):
        angle = 2 * np.pi * i / (num_points * 2)
        r = radius if i % 2 == 0 else radius / 2
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        points.append((int(x), int(y)))
    cv2.fillPoly(img, [np.array(points)], (0, 255, 255))
    
    return img

# Generate and save the original image
original_img = create_sample_image()
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.tight_layout()
plt.savefig('static/images/cv/original_image.png', dpi=150)
plt.close()

print("Original image saved.")

# 2. Create grayscale version
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(8, 6))
plt.imshow(gray_img, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.tight_layout()
plt.savefig('static/images/cv/grayscale_image.png', dpi=150)
plt.close()

print("Grayscale image saved.")

# 3. Create blurred version
blur_img = cv2.GaussianBlur(original_img, (15, 15), 0)
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
plt.title('Blurred Image')
plt.axis('off')
plt.tight_layout()
plt.savefig('static/images/cv/blurred_image.png', dpi=150)
plt.close()

print("Blurred image saved.")

# 4. Create edge detection example
edges = cv2.Canny(gray_img, 50, 150)
plt.figure(figsize=(8, 6))
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.tight_layout()
plt.savefig('static/images/cv/canny_edges.png', dpi=150)
plt.close()

print("Edge detection image saved.")

# 5. Create object detection example
def create_object_detection_example():
    # Create a street scene with cars and people
    img = np.ones((500, 700, 3), dtype=np.uint8) * 200  # Light gray background for road
    
    # Add a blue sky
    cv2.rectangle(img, (0, 0), (700, 150), (230, 180, 50), -1)
    
    # Add green grass at the bottom
    cv2.rectangle(img, (0, 450), (700, 500), (60, 170, 80), -1)
    
    # Add some buildings
    cv2.rectangle(img, (50, 50), (150, 200), (90, 90, 90), -1)  # Building 1
    cv2.rectangle(img, (200, 70), (300, 180), (110, 110, 120), -1)  # Building 2
    cv2.rectangle(img, (400, 30), (550, 190), (70, 70, 90), -1)  # Building 3
    
    # Add some simple "cars"
    cv2.rectangle(img, (100, 320), (200, 380), (0, 0, 255), -1)  # Red car
    cv2.rectangle(img, (400, 340), (520, 400), (255, 0, 0), -1)  # Blue car
    
    # Add some simple "people"
    # Person 1 (simplified as a small rectangle with a circle on top)
    cv2.rectangle(img, (250, 400), (270, 440), (180, 120, 100), -1)  # Body
    cv2.circle(img, (260, 390), 15, (180, 120, 100), -1)  # Head
    
    # Person 2
    cv2.rectangle(img, (600, 380), (620, 420), (100, 150, 200), -1)  # Body
    cv2.circle(img, (610, 370), 15, (100, 150, 200), -1)  # Head
    
    # Draw bounding boxes and labels
    # Red car
    cv2.rectangle(img, (95, 315), (205, 385), (0, 255, 0), 2)
    cv2.putText(img, "Car", (110, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Blue car
    cv2.rectangle(img, (395, 335), (525, 405), (0, 255, 0), 2)
    cv2.putText(img, "Car", (430, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Person 1
    cv2.rectangle(img, (245, 385), (275, 445), (0, 255, 0), 2)
    cv2.putText(img, "Person", (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Person 2
    cv2.rectangle(img, (595, 365), (625, 425), (0, 255, 0), 2)
    cv2.putText(img, "Person", (550, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img

# Create and save object detection example
obj_detect_img = create_object_detection_example()
plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(obj_detect_img, cv2.COLOR_BGR2RGB))
plt.title('Object Detection Example')
plt.axis('off')
plt.tight_layout()
plt.savefig('static/images/cv/object_detection.png', dpi=150)
plt.close()

print("Object detection example saved.")

# 6. Create segmentation example
def create_segmentation_example():
    # Create an image similar to the object detection but with segmentation masks
    base_img = create_object_detection_example()
    
    # Create a blank mask image
    mask = np.zeros(base_img.shape, dtype=np.uint8)
    
    # Red car segmentation (red)
    cv2.rectangle(mask, (100, 320), (200, 380), (0, 0, 255), -1)
    
    # Blue car segmentation (blue)
    cv2.rectangle(mask, (400, 340), (520, 400), (255, 0, 0), -1)
    
    # Person 1 segmentation (green)
    cv2.rectangle(mask, (250, 400), (270, 440), (0, 255, 0), -1)
    cv2.circle(mask, (260, 390), 15, (0, 255, 0), -1)
    
    # Person 2 segmentation (yellow)
    cv2.rectangle(mask, (600, 380), (620, 420), (0, 255, 255), -1)
    cv2.circle(mask, (610, 370), 15, (0, 255, 255), -1)
    
    # Road segmentation (gray)
    road_mask = np.zeros_like(mask)
    cv2.rectangle(road_mask, (0, 150), (700, 450), (128, 128, 128), -1)
    # Remove the objects from road
    road_mask = np.where(mask > 0, 0, road_mask)
    mask = mask + road_mask
    
    # Buildings segmentation (dark gray)
    cv2.rectangle(mask, (50, 50), (150, 200), (90, 90, 90), -1)
    cv2.rectangle(mask, (200, 70), (300, 180), (90, 90, 90), -1)
    cv2.rectangle(mask, (400, 30), (550, 190), (90, 90, 90), -1)
    
    # Sky segmentation (light blue)
    sky_mask = np.zeros_like(mask)
    cv2.rectangle(sky_mask, (0, 0), (700, 150), (230, 180, 50), -1)
    # Remove the buildings from sky
    for y in range(sky_mask.shape[0]):
        for x in range(sky_mask.shape[1]):
            if np.array_equal(mask[y, x], [90, 90, 90]):
                sky_mask[y, x] = [0, 0, 0]
    mask = np.where(sky_mask > 0, sky_mask, mask)
    
    # Grass segmentation (green)
    cv2.rectangle(mask, (0, 450), (700, 500), (60, 170, 80), -1)
    
    # Create a blended visualization
    alpha = 0.5
    blended = cv2.addWeighted(base_img, 1-alpha, mask, alpha, 0)
    
    # Create side-by-side visualization
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax[0].imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Segmentation mask
    ax[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Segmentation Mask')
    ax[1].axis('off')
    
    # Blended visualization
    ax[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    ax[2].set_title('Blended Visualization')
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('static/images/cv/segmentation.png', dpi=150)
    plt.close()

# Create and save segmentation example
create_segmentation_example()
print("Segmentation example saved.")

# 7. Create a face recognition example
def create_face_recognition_example():
    # Create a simple image with cartoon faces
    img = np.ones((700, 900, 3), dtype=np.uint8) * 245  # Lighter background and larger canvas
    
    # Add title
    cv2.putText(img, "Face Recognition Pipeline", (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Draw 3 simple cartoon faces - make them more distinct
    # Face 1
    cv2.circle(img, (150, 150), 65, (220, 220, 220), -1)  # Face
    cv2.circle(img, (130, 135), 13, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (170, 135), 13, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(img, (150, 165), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Smile
    cv2.line(img, (110, 110), (130, 90), (0, 0, 0), 2)  # Left eyebrow
    cv2.line(img, (170, 90), (190, 110), (0, 0, 0), 2)  # Right eyebrow
    
    # Face 2 (similar to Face 1 - same person)
    cv2.circle(img, (400, 150), 65, (220, 220, 220), -1)  # Face
    cv2.circle(img, (380, 135), 13, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (420, 135), 13, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(img, (400, 170), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Smile - slightly different
    cv2.line(img, (360, 110), (380, 90), (0, 0, 0), 2)  # Left eyebrow
    cv2.line(img, (420, 90), (440, 110), (0, 0, 0), 2)  # Right eyebrow
    
    # Face 3 (different person)
    cv2.circle(img, (650, 150), 65, (220, 220, 220), -1)  # Face
    cv2.circle(img, (630, 135), 13, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (670, 135), 13, (0, 0, 0), -1)  # Right eye
    cv2.line(img, (625, 170), (675, 170), (0, 0, 0), 2)  # Neutral mouth
    cv2.line(img, (610, 115), (630, 95), (0, 0, 0), 2)  # Left eyebrow
    cv2.line(img, (670, 95), (690, 115), (0, 0, 0), 2)  # Right eyebrow
    
    # Add facial landmarks
    # Face 1 landmarks
    cv2.circle(img, (130, 135), 3, (255, 0, 0), -1)  # Left eye center
    cv2.circle(img, (170, 135), 3, (255, 0, 0), -1)  # Right eye center
    cv2.circle(img, (150, 160), 3, (255, 0, 0), -1)  # Nose tip
    cv2.circle(img, (130, 180), 3, (255, 0, 0), -1)  # Left mouth corner
    cv2.circle(img, (170, 180), 3, (255, 0, 0), -1)  # Right mouth corner
    
    # Face 2 landmarks
    cv2.circle(img, (380, 135), 3, (255, 0, 0), -1)  # Left eye center
    cv2.circle(img, (420, 135), 3, (255, 0, 0), -1)  # Right eye center
    cv2.circle(img, (400, 160), 3, (255, 0, 0), -1)  # Nose tip
    cv2.circle(img, (380, 185), 3, (255, 0, 0), -1)  # Left mouth corner
    cv2.circle(img, (420, 185), 3, (255, 0, 0), -1)  # Right mouth corner
    
    # Face 3 landmarks
    cv2.circle(img, (630, 135), 3, (255, 0, 0), -1)  # Left eye center
    cv2.circle(img, (670, 135), 3, (255, 0, 0), -1)  # Right eye center
    cv2.circle(img, (650, 160), 3, (255, 0, 0), -1)  # Nose tip
    cv2.circle(img, (630, 170), 3, (255, 0, 0), -1)  # Left mouth corner
    cv2.circle(img, (670, 170), 3, (255, 0, 0), -1)  # Right mouth corner
    
    # Add workflow elements
    # Draw arrows between steps
    cv2.arrowedLine(img, (150, 230), (150, 290), (50, 50, 50), 2, tipLength=0.03)
    cv2.arrowedLine(img, (400, 230), (400, 290), (50, 50, 50), 2, tipLength=0.03)
    cv2.arrowedLine(img, (650, 230), (650, 290), (50, 50, 50), 2, tipLength=0.03)
    
    cv2.arrowedLine(img, (150, 370), (150, 430), (50, 50, 50), 2, tipLength=0.03)
    cv2.arrowedLine(img, (400, 370), (400, 430), (50, 50, 50), 2, tipLength=0.03)
    cv2.arrowedLine(img, (650, 370), (650, 430), (50, 50, 50), 2, tipLength=0.03)
    
    cv2.arrowedLine(img, (150, 510), (150, 570), (50, 50, 50), 2, tipLength=0.03)
    cv2.arrowedLine(img, (400, 510), (400, 570), (50, 50, 50), 2, tipLength=0.03)
    cv2.arrowedLine(img, (650, 510), (650, 570), (50, 50, 50), 2, tipLength=0.03)
    
    # Step 1: Detection - add clear step labels
    cv2.rectangle(img, (30, 230), (870, 290), (240, 240, 240), -1)
    cv2.putText(img, "STEP 1: Face Detection", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Detection boxes
    cv2.rectangle(img, (85, 85), (215, 215), (0, 200, 0), 2)
    cv2.rectangle(img, (335, 85), (465, 215), (0, 200, 0), 2)
    cv2.rectangle(img, (585, 85), (715, 215), (0, 200, 0), 2)
    
    # Step 2: Landmark Detection
    cv2.rectangle(img, (30, 310), (870, 370), (240, 240, 240), -1)
    cv2.putText(img, "STEP 2: Facial Landmark Detection", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Step 3: Alignment
    cv2.rectangle(img, (30, 430), (870, 490), (240, 240, 240), -1)
    cv2.putText(img, "STEP 3: Face Alignment", (50, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Draw alignment lines and rotated faces
    cv2.line(img, (120, 135), (180, 135), (0, 0, 255), 2)  # Horizontal eye line for face 1
    cv2.line(img, (370, 135), (430, 135), (0, 0, 255), 2)  # Horizontal eye line for face 2
    cv2.line(img, (620, 135), (680, 135), (0, 0, 255), 2)  # Horizontal eye line for face 3
    
    # Step 4: Feature Extraction
    cv2.rectangle(img, (30, 550), (870, 610), (240, 240, 240), -1)
    cv2.putText(img, "STEP 4: Feature Extraction & Matching", (50, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Better visualization for feature vectors
    # Face 1 feature vector
    feature_bg = np.ones((40, 140, 3), dtype=np.uint8) * 245
    for i in range(10):
        val = np.random.randint(30, 100) if i < 7 else np.random.randint(5, 25)
        cv2.line(feature_bg, (10 + i*12, 30), (10 + i*12, 30-val), (0, 0, 180), 4)
    feature_bg_1 = feature_bg.copy()
    img[500:540, 80:220] = feature_bg_1
    cv2.rectangle(img, (80, 500), (220, 540), (0, 0, 0), 1)
    cv2.putText(img, "Feature Vector A", (85, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Face 2 feature vector (similar to Face 1 - same person)
    feature_bg = np.ones((40, 140, 3), dtype=np.uint8) * 245
    for i in range(10):
        val = np.random.randint(25, 95) if i < 7 else np.random.randint(5, 25)
        cv2.line(feature_bg, (10 + i*12, 30), (10 + i*12, 30-val), (0, 0, 180), 4)
    feature_bg_2 = feature_bg.copy()
    img[500:540, 330:470] = feature_bg_2
    cv2.rectangle(img, (330, 500), (470, 540), (0, 0, 0), 1)
    cv2.putText(img, "Feature Vector A'", (335, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Face 3 feature vector (different pattern - different person)
    feature_bg = np.ones((40, 140, 3), dtype=np.uint8) * 245
    for i in range(10):
        val = np.random.randint(5, 25) if i < 7 else np.random.randint(30, 100)
        cv2.line(feature_bg, (10 + i*12, 30), (10 + i*12, 30-val), (0, 0, 180), 4)
    feature_bg_3 = feature_bg.copy()
    img[500:540, 580:720] = feature_bg_3
    cv2.rectangle(img, (580, 500), (720, 540), (0, 0, 0), 1)
    cv2.putText(img, "Feature Vector B", (585, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add comparison visualization
    # Draw comparison arrows
    cv2.arrowedLine(img, (150, 550), (150, 610), (50, 50, 50), 2, tipLength=0.03)
    cv2.arrowedLine(img, (400, 550), (400, 610), (50, 50, 50), 2, tipLength=0.03)
    cv2.arrowedLine(img, (650, 550), (650, 610), (50, 50, 50), 2, tipLength=0.03)
    
    # Draw similarity calculations with clear visual indicators
    # Match between Face 1 and Face 2
    cv2.line(img, (180, 620), (320, 620), (0, 180, 0), 2)
    cv2.circle(img, (250, 640), 20, (200, 255, 200), -1)
    cv2.circle(img, (250, 640), 20, (0, 150, 0), 2)
    cv2.putText(img, "92%", (235, 645), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)
    cv2.putText(img, "MATCH (Same Person)", (160, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 130, 0), 2)
    
    # Non-match between Face 2 and Face 3
    cv2.line(img, (480, 620), (620, 620), (180, 0, 0), 2)
    cv2.circle(img, (550, 640), 20, (255, 220, 220), -1)
    cv2.circle(img, (550, 640), 20, (150, 0, 0), 2)
    cv2.putText(img, "28%", (535, 645), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 0), 2)
    cv2.putText(img, "NO MATCH (Different Person)", (450, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 0, 0), 2)
    
    # Draw connection lines to better show the comparison flow
    # Line connecting face 1 to face 2 comparison
    cv2.line(img, (150, 640), (230, 640), (0, 150, 0), 2, cv2.LINE_AA)
    cv2.line(img, (270, 640), (350, 640), (0, 150, 0), 2, cv2.LINE_AA)
    
    # Line connecting face 2 to face 3 comparison
    cv2.line(img, (450, 640), (530, 640), (150, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img, (570, 640), (650, 640), (150, 0, 0), 2, cv2.LINE_AA)
    
    # Add IDs to the faces
    cv2.putText(img, "ID: Person A", (110, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 200), 2)
    cv2.putText(img, "ID: Person A", (360, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 200), 2)
    cv2.putText(img, "ID: Person B", (610, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 200), 2)
    
    return img

# Create and save face recognition example
face_rec_img = create_face_recognition_example()
plt.figure(figsize=(14, 11))
plt.imshow(cv2.cvtColor(face_rec_img, cv2.COLOR_BGR2RGB))
plt.title('Face Recognition Pipeline', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.savefig('static/images/cv/face_recognition.png', dpi=200, bbox_inches='tight')
plt.close()

print("Face recognition example saved.")

print("All Computer Vision example images have been generated successfully.") 