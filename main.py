from ultralytics import YOLO
import cv2 as cv

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load video
video_path = './test.mp4'
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read and process frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Detect and track objects
    results = model.track(frame, persist=True, verbose=False)  # `persist` retains tracking IDs

    # Plot results on the frame
    frame_with_results = results[0].plot()  # Get annotated frame

    # Visualize
    cv.imshow('YOLOv8 Tracking', frame_with_results)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
