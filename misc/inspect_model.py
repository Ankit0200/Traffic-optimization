
from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('last.pt')

# Print classes
print("Model classes:")
print(model.names)

# Open video
cap = cv2.VideoCapture('4.mp4')
ret, frame = cap.read()
if not ret:
    print("Error reading video 4.mp4")
else:
    # Run inference
    results = model(frame)
    
    # Process results
    for result in results:
        boxes = result.boxes
        print(f"Detected {len(boxes)} objects")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"Class: {model.names[cls_id]}, Confidence: {conf:.2f}")

    # Visualize
    res_plotted = results[0].plot()
    cv2.imwrite("model_check_result.jpg", res_plotted)
    print("Saved inference result to model_check_result.jpg")

cap.release()
