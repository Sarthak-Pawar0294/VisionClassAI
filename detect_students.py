from ultralytics import YOLO
import cv2
import torch

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    # Run YOLO
    results = model(
        frame,
        device=device,
        classes=[0],       # class 0 = person
        conf=0.40,
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Count detected people
    person_count = len(results[0].boxes)

    cv2.putText(
        annotated_frame,
        f"Students Detected: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("VisionClass - Student Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()