import cv2


def applyCartoonFilter(frame):
    # 1. Convert to grayscale and blur to remove image noise
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    
    # 2. Use adaptive thresholding to detect and draw comic book edges
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)
    
    # 3. Apply a bilateral filter to drastically smooth skin and flatten colors
    color_smooth = cv2.bilateralFilter(frame, 9, 250, 250)
    
    # 4. Combine the cartoon edges with the smooth color palette
    cartoon_filter = cv2.bitwise_and(color_smooth, color_smooth, mask=edges)
    return cartoon_filter
    
   
    

cap = cv2.VideoCapture(0)
print("Controls: 'g' = Grayscale, 'b' = Blur, 'e' = Canny Edges, 'c' = Cartoon, 'n' = Normal, 'q' = Quit")
filter_mode = 'normal'

ret, frame = cap.read() # get first frame

if not ret:
    print("Could not access camera")
    cap.release()
    exit()


# Select ROI using mouse
x, y, w, h = cv2.selectROI("Select Region", frame, False)

cv2.destroyWindow("Select Region")

# Check if user cancelled selection
if w == 0 or h == 0:
    print("No region selected")
    cap.release()
    cv2.destroyAllWindows()
    exit()


while True:
   
   # IMPORTANT: Get a fresh frame every iteration
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break
    
    # Make sure ROI is within the frame
    roi = frame[y:y+h, x:x+w]      
    
     # Apply selected filter to the selected frame(roi)
    if filter_mode == 'grayscale':
        # Convert ROI to grayscale
        edited_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

        # Convert back to BGR so it fits into the frame
        edited_roi = cv2.cvtColor(edited_roi,cv2.COLOR_GRAY2BGR)

    elif filter_mode == 'blur':
        # Applies a Gaussian Blur filter
        edited_roi = cv2.GaussianBlur(roi, (15, 15), 0)
        
    elif filter_mode == 'canny':
        # Canny produces a grayscale image
        edited_roi = cv2.Canny(roi,30,150)

        # Convert back to BGR
        edited_roi = cv2.cvtColor(edited_roi,cv2.COLOR_GRAY2BGR)
        
    elif filter_mode == 'cartoon':
        edited_roi = applyCartoonFilter(roi)
        
    else:
        # Default, unmodified live video
        edited_roi = roi
        
    # Put filtered ROI back into frame
    frame[y:y+h, x:x+w] = edited_roi
    
    
    # Display
    cv2.imshow("Camera", frame)
    
    
    # Listen for key strokes to switch filters
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('g'):
        filter_mode = 'grayscale'        
    elif key == ord('b'):
        filter_mode = 'blur'
    elif key == ord('e'):
        filter_mode = 'canny'
    elif key == ord('n'):
        filter_mode = 'normal'
    elif key == ord('c'):
        filter_mode = 'cartoon'

    if key == ord("q"):
        break
    
   

cap.release()
cv2.destroyAllWindows()
