import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math

# =============================================================================
# 1. SETUP & CONFIGURATION (Based on Source Section 3.1)
# =============================================================================

# Load YOLO
model = YOLO('10_epoch.pt') # Use nano for speed, or your '10_epoch.pt'
class_list = model.names

# Open Video
cap = cv2.VideoCapture('new.mp4')

# --- HOMOGRAPHY MATRIX (Placeholder) ---
# In reality, you calculate this using 4 known points on the road.
# This matrix maps [pixel_x, pixel_y, 1] -> [meters_x, meters_y, 1]
# This example matrix assumes the camera is looking down at an angle.
H_matrix = np.array([
    [ 1.5e-02, -5.0e-03, -1.0e+01],
    [ 2.0e-04,  4.0e-02, -5.0e+01],
    [ 1.0e-05, -2.0e-04,  1.0e+00]
])

def pixel_to_meters(px, py, H):
    """
    Converts (u,v) pixels to (X,Y) meters using Homography.
    Source: Section 3.1 "x proportional to H p"
    Homogeneous result is (x', y', w); we divide by w to get real-world coords.
    """
    point = np.array([px, py, 1], dtype=np.float32)
    new_point = np.dot(H, point)
    # Perspective divide: use 3rd component (w), not 2nd
    w = new_point[2]
    if abs(w) < 1e-9:
        w = 1e-9  # avoid division by zero
    x = new_point[0] / w
    y = new_point[1] / w
    return x, y

# --- GRID CONFIGURATION (Your Request) ---
# We divide the world into 2x2 meter blocks, not pixels.
BLOCK_SIZE_METERS = 2.0 

# Data Storage: Map Block ID -> List of Trajectories
# Key: (Block_X, Block_Y) e.g., (5, 12)
# Value: List of speeds/headings observed in this block
grid_data = defaultdict(list)

# Track histories for velocity calculation
# ID -> deque of last 30 frames (x_meter, y_meter, timestamp)
track_history = defaultdict(lambda: deque(maxlen=30)) 

# =============================================================================
# 2. MAIN LOOP
# =============================================================================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO tracking
    results = model.track(frame, persist=True, classes=[0], verbose=False) # Classes: Car, motorcycle, bus, truck

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # 1. Get Center Point (Pixels)
            cx_pixel = (x1 + x2) // 2
            cy_pixel = (y1 + y2) // 2 # Use bottom of car for better accuracy
            
            # 2. Convert to Meters (Ground Plane) [Source: 3.1]
            mx, my = pixel_to_meters(cx_pixel, cy_pixel, H_matrix)
            
            # 3. Calculate Velocity & Heading [Source: 3.3]
            velocity = 0.0
            heading = 0.0
            history = track_history[track_id]
            
            # Add current point to history
            history.append((mx, my))
            
            if len(history) > 1:
                # Simple velocity: dist / time (assuming 30fps, dt ~ 0.033s)
                prev_mx, prev_my = history[-2]
                dx = mx - prev_mx
                dy = my - prev_my
                dist = math.sqrt(dx**2 + dy**2)
                velocity = dist * 30 # approx m/s
                heading = math.degrees(math.atan2(dy, dx))

            # 4. Assign to Grid Block (Your Request)
            # Map meter coordinate to an integer block ID
            block_x = int(mx // BLOCK_SIZE_METERS)
            block_y = int(my // BLOCK_SIZE_METERS)
            block_id = (block_x, block_y)
            
            # 5. Collect Data for this Block
            # We store: "Car #104 was in Block (5,12) moving at 15m/s angled 90 deg"
            grid_data[block_id].append({
                "id": track_id,
                "velocity": round(velocity, 2),
                "heading": round(heading, 2)
            })

            # --- VISUALIZATION ---
            
            # Draw the car center
            cv2.circle(frame, (cx_pixel, cy_pixel), 5, (0, 0, 255), -1)
            
            # Draw the Grid Block info on screen
            # We project the "Block ID" back to text
            info_text = f"ID:{track_id} Blk:{block_id}"
            stats_text = f"V:{velocity:.1f}m/s"
            
            cv2.putText(frame, info_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, stats_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Visualization: Draw a representation of the grid (Optional)
    # This just shows the user that "Meters" are working
    cv2.putText(frame, "Grid: 2x2 Meters (Active)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Tracking with Metric Grid", frame)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
