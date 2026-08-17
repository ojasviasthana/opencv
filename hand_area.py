import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# --------------------------------------------------
# Camera
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Camera started.")
print("Press Q to quit.")

frame_timestamp = 0

# --------------------------------------------------
# Hand landmark detector
# --------------------------------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # --------------------------------------------------
        # OpenCV BGR -> MediaPipe RGB
        # --------------------------------------------------

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # --------------------------------------------------
        # Detect hands
        # --------------------------------------------------

        frame_timestamp += 33

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        # --------------------------------------------------
        # We need TWO hands
        # --------------------------------------------------

        if len(result.hand_landmarks) == 2:

            hand1 = result.hand_landmarks[0]
            hand2 = result.hand_landmarks[1]

            # ----------------------------------------------
            # Get wrist positions
            # Landmark 0 = wrist
            # ----------------------------------------------

            wrist1 = hand1[12]
            wrist2 = hand2[12]

            x1 = int(wrist1.x * width)
            y1 = int(wrist1.y * height)

            x2 = int(wrist2.x * width)
            y2 = int(wrist2.y * height)

            # ----------------------------------------------
            # Make left/right points
            # ----------------------------------------------

            left_x = min(x1, x2)
            right_x = max(x1, x2)

            top_y = min(y1, y2)
            bottom_y = max(y1, y2)

            # ----------------------------------------------
            # Draw line between wrists
            # ----------------------------------------------

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # ----------------------------------------------
            # Rectangle between hands
            # ----------------------------------------------

            cv2.rectangle(
                frame,
                (left_x, top_y),
                (right_x, bottom_y),
                (255, 0, 0),
                3
            )

            # ----------------------------------------------
            # Calculate area
            # ----------------------------------------------

            rectangle_width = right_x - left_x
            rectangle_height = bottom_y - top_y

            area = rectangle_width * rectangle_height

            # ----------------------------------------------
            # Draw transparent area
            # ----------------------------------------------

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (left_x, top_y),
                (right_x, bottom_y),
                (255, 0, 0),
                -1
            )

            frame = cv2.addWeighted(
                overlay,
                0.2,
                frame,
                0.8,
                0
            )

            # Redraw border
            cv2.rectangle(
                frame,
                (left_x, top_y),
                (right_x, bottom_y),
                (255, 0, 0),
                3
            )

            # ----------------------------------------------
            # Display area
            # ----------------------------------------------

            cv2.putText(
                frame,
                f"Area: {area} px^2",
                (left_x, max(top_y - 15, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Width: {rectangle_width}px",
                (left_x, bottom_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Height: {rectangle_height}px",
                (left_x, bottom_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "Show BOTH hands",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # --------------------------------------------------
        # Show camera
        # --------------------------------------------------

        cv2.imshow("Area Between Hands", frame)

        # Q = quit
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
cv2.destroyAllWindows()