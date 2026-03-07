import cv2
from ultralytics import YOLO
from collections import defaultdict


# Load the YOLO model
model = YOLO('10_epoch.pt')

class_list = model.names 

# Open the video file
cap = cv2.VideoCapture('control.mp4')

# NEW: Get frame dimensions for grid setup
ret, first_frame = cap.read()
if ret:
    frame_height, frame_width = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    print(f"Frame size: {frame_width}x{frame_height}")

# NEW: Grid cell configuration
CELL_SIZE = 30  # pixels per cell (adjust based on your needs: 5, 10, 20, etc.)
SHOW_GRID = True  # Set to False to hide grid lines

# NEW: Function to convert pixel coordinates to grid cell
def pixel_to_cell(x, y, cell_size=CELL_SIZE):
    """
    Convert pixel coordinates to grid cell ID
    Args:
        x, y: pixel coordinates
        cell_size: size of each cell in pixels
    Returns:
        (cell_x, cell_y): grid cell coordinates
    """
    cell_x = int(x // cell_size)
    cell_y = int(y // cell_size)
    return (cell_x, cell_y)

# NEW: Function to draw grid overlay (optional visualization)
def draw_grid(frame, cell_size=CELL_SIZE, color=(50, 50, 50), thickness=1):
    """
    Draw grid lines on frame for visualization
    Args:
        frame: video frame
        cell_size: size of cells in pixels
        color: grid line color (BGR)
        thickness: line thickness
    """
    height, width = frame.shape[:2]
    
    # Draw vertical lines
    for x in range(0, width, cell_size):
        cv2.line(frame, (x, 0), (x,height), color, thickness)
    
    # Draw horizontal lines
    for y in range(0, height, cell_size):
        cv2.line(frame, (0, y), (width, y), color, thickness)
    
    return frame

# NEW: Dictionary to store trajectory history per track_id
# Format: {track_id: [(x, y, frame_num, cell_x, cell_y), ...]}
trajectory_history = defaultdict(list)

# NEW: Frame counter for trajectory timestamps
frame_number = 0

line_y_red = 430  # Red line position

# Dictionary to store object counts by class
class_counts = defaultdict(int)

# Dictionary to keep track of object IDs that have crossed the line
crossed_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # NEW: Increment frame counter
    frame_number += 1

    # NEW: Draw grid overlay (optional - helps visualize cells)
    if SHOW_GRID:
        frame = draw_grid(frame, CELL_SIZE)

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes = [0]) 
    print(results)

    # Ensure tracked objects exist (otherwise .id is None)
    if results[0].boxes.id is not None:
        # Get the detected boxes, their class indices, and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        cv2.line(frame,(100,100),(200,200),(0,255,0),2)

        # Loop through each detected object
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2  # Calculate the center point
            cy = (y1 + y2) // 2            

            class_name = class_list[class_idx]

            # NEW: Convert center point to grid cell
            cell_x, cell_y = pixel_to_cell(cx, cy, CELL_SIZE)
            
            # NEW: Store trajectory with cell in cformation
            trajectory_history[track_id].append((cx, cy, frame_number, cell_x, cell_y))
            
            # NEW: Display cell coordinates on vehicle (optional visualization)
            cv2.putText(frame, f"Cell: ({cell_x},{cell_y})", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 

    # NEW: Display grid information on frame
    cv2.putText(frame, f"Cell Size: {CELL_SIZE}px | Frame: {frame_number}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Tracked Objects: {len(trajectory_history)}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show the frame
    cv2.imshow("YOLO Object Tracking & Counting", frame)    
    
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# NEW: Print trajectory summary (optional - see what was captured)
print("\n=== Trajectory Summary ===")
for track_id, trajectory in trajectory_history.items():
    print(f"Track ID {track_id}: {len(trajectory)} positions recorded")
    # Print first and last cell position
    if len(trajectory) > 0:
        first_cell = trajectory[0][3:5]  # (cell_x, cell_y)
        last_cell = trajectory[-1][3:5]
        print(f"  Started at cell {first_cell}, ended at cell {last_cell}")
