import cv2
import numpy as np

def count_fingers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    finger_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:  # Adjust the threshold as needed
            hull = cv2.convexHull(contour)
            hull_points = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull_points)
            
            if defects is not None:
                count = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    
                    a = np.linalg.norm(np.array(start) - np.array(end))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(far) - np.array(end))
                    
                    angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
                    
                    if angle <= np.pi / 2 and d > 10000:
                        count += 1
                
                finger_count = count + 1
    
    return finger_count

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


