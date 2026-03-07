import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load the YOLO model
model = YOLO('10_epoch.pt')

class_list = model.names 
#class_list

# Open the video file
cap = cv2.VideoCapture('new.mp4')

line_y_red = 430  # Red line position

# Dictionary to store object counts by class
class_counts = defaultdict(int)

# Dictionary to keep track of object IDs that have crossed the line
crossed_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes = [0]) 
    #print(results)
    print(results)

    # Ensure tracked objects exist (otherwise .id is None)
    if results[0].boxes.id is not None:
        # Get the detected boxes, their class indices, and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        cv2.line(frame,(100,100),(200,200),(0,255,0),2)
        #cv2.putText(frame, 'Red Line', (690, line_y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        

        # Loop through each detected object
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2  # Calculate the center point
            cy = (y1 + y2) // 2            

            class_name = class_list[class_idx]

            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 


        #     # Check if the object has crossed the red line
        #     if cy > line_y_red and track_id not in crossed_ids:
        #         # Mark the object as crossed
        #         crossed_ids.add(track_id)
        #         class_counts[class_name] += 1


        # # Display the counts on the frame
        # y_offset = 30

        # for class_name, count in class_counts.items():
        #     cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        #     y_offset += 30

    
    
    # Show the frame

    cv2.imshow("YOLO Object Tracking & Counting", frame)    
    
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(0 ) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()