import cv2
import numpy as np

def count_fingers(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Threshold the HSV image to get only skin colors
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Find contours
    contours, _ = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0
    
    # Find the largest contour
    max_contour = max(contours, key=cv2.contourArea)
    
    # Find the convex hull of the largest contour
    hull = cv2.convexHull(max_contour)
    
    # Find convexity defects
    hull_indices = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull_indices)
    
    if defects is None:
        return 0
    
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])
        
        # Calculate the angle between the fingers
        a = np.linalg.norm(np.array(start) - np.array(end))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(far) - np.array(end))
        
        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
        
        # If angle is less than 90 degrees, it's considered a finger
        if angle <= np.pi / 2 and d > 10000:
            finger_count += 1
    
    return finger_count + 1

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    num_fingers = count_fingers(frame)

    cv2.putText(frame, f'Number of Fingers: {num_fingers}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


