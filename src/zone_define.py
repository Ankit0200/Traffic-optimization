"""
Exit Line Setup Tool
====================
Usage:
    python setup_zones.py --video control.mp4
    python setup_zones.py --video control.mp4 --output my_config.json

Controls:
    Left Click (1st) : Set start point of a line
    Left Click (2nd) : Set end point of a line → prompts for label
    'u'              : Undo last saved line
    'r'              : Reset all lines
    's'              : Save config to JSON and quit
    'q' / ESC        : Quit without saving

Workflow:
    1. Run this script with your video file
    2. A frame appears — click two points to define an exit line
    3. Type a label in the terminal (e.g., "left", "through", "right")
    4. Repeat for all exit directions
    5. Press 's' to save → produces a JSON config file
    6. Your main pipeline loads that JSON and detects line crossings
"""

import cv2
import json
import argparse
import numpy as np
from pathlib import Path


# ── State ────────────────────────────────────────────────────────────────
first_point = None       # First click of current line
saved_lines = []         # List of {"label": str, "line": [[x1,y1], [x2,y2]]}
frame_clean = None       # Original frame (no drawings)
frame_display = None     # Frame with drawings
window_name = "Exit Line Setup — Click 2 points per line | 's': save | 'q': quit"


# ── Colors (cycles through these) ────────────────────────────────────────
COLORS = [
    (0, 0, 255),     # Red
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (255, 165, 0),   # Orange
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
]


def get_color(idx):
    return COLORS[idx % len(COLORS)]


def redraw():
    """Redraw everything on the frame."""
    global frame_display
    frame_display = frame_clean.copy()

    # Draw all saved lines
    for i, line_data in enumerate(saved_lines):
        color = get_color(i)
        p1, p2 = line_data["line"]
        cv2.line(frame_display, tuple(p1), tuple(p2), color, 3)

        # Draw endpoints
        cv2.circle(frame_display, tuple(p1), 6, color, -1)
        cv2.circle(frame_display, tuple(p2), 6, color, -1)

        # Draw direction arrow (small arrow at midpoint showing which side is "crossed")
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2

        # Label at midpoint
        cv2.putText(frame_display, f'{line_data["label"]}',
                    (mx - 20, my - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_display, f'{line_data["label"]}',
                    (mx - 20, my - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)

    # Draw first point of current line (waiting for second click)
    if first_point is not None:
        cv2.circle(frame_display, first_point, 8, (0, 255, 255), -1)
        cv2.putText(frame_display, "Click 2nd point...",
                    (first_point[0] + 10, first_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # HUD
    status = "Click 1st point" if first_point is None else "Click 2nd point to finish line"
    cv2.putText(frame_display, f"Saved lines: {len(saved_lines)} | {status}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame_display, "L-click: define line | 'u': undo | 'r': reset | 's': save | 'q': quit",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Show saved line list
    for i, line_data in enumerate(saved_lines):
        p1, p2 = line_data["line"]
        cv2.putText(frame_display,
                    f"{i}: {line_data['label']} ({p1[0]},{p1[1]})->({p2[0]},{p2[1]})",
                    (10, 80 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, get_color(i), 1)

    cv2.imshow(window_name, frame_display)


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks."""
    global first_point

    if event == cv2.EVENT_LBUTTONDOWN:
        if first_point is None:
            # First click — set start point
            first_point = (x, y)
            print(f"  Start point: ({x}, {y}) — now click the end point")
            redraw()
        else:
            # Second click — complete the line
            second_point = (x, y)
            print(f"  End point: ({x}, {y})")

            # Check that points aren't too close
            dist = np.sqrt((first_point[0] - x)**2 + (first_point[1] - y)**2)
            if dist < 10:
                print("  ⚠ Points too close together. Try again.")
                first_point = None
                redraw()
                return

            # Prompt for label
            label = input("  Enter label for this line (e.g., left, through, right): ").strip()
            if not label:
                label = f"exit_{len(saved_lines)}"

            saved_lines.append({
                "label": label,
                "line": [list(first_point), list(second_point)]
            })
            print(f"  ✓ Line '{label}' saved!\n")

            first_point = None
            redraw()


def main():
    global frame_clean, first_point

    parser = argparse.ArgumentParser(description="Interactive Exit Line Setup Tool")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default=None, help="Output JSON path (default: <video_name>_config.json)")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to use (default: 0)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.with_name(f"{video_path.stem}_config.json")

    # Load the frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video '{args.video}'")
        return

    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    ret, frame_clean = cap.read()
    cap.release()

    if not ret:
        print("Error: Cannot read frame from video.")
        return

    h, w = frame_clean.shape[:2]
    print(f"\n{'='*50}")
    print(f"  Exit Line Setup Tool")
    print(f"  Video: {args.video}")
    print(f"  Frame size: {w} x {h}")
    print(f"  Output: {output_path}")
    print(f"{'='*50}")
    print(f"\n  Instructions:")
    print(f"    Left-click twice → define a line (start → end)")
    print(f"    Type a label when prompted (left / through / right)")
    print(f"    'u' → undo last line")
    print(f"    'r' → reset everything")
    print(f"    's' → save to JSON and quit")
    print(f"    'q' / ESC → quit without saving\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == ord('u'):
            if saved_lines:
                removed = saved_lines.pop()
                print(f"  Undo: removed line '{removed['label']}'")
                redraw()
            elif first_point is not None:
                first_point = None
                print("  Undo: cleared start point")
                redraw()
            else:
                print("  Nothing to undo.")

        elif key == ord('r'):
            first_point = None
            saved_lines.clear()
            print("  ✓ All lines reset.")
            redraw()

        elif key == ord('s'):
            if not saved_lines:
                print("  ⚠ No lines defined yet. Define at least one line before saving.")
                continue

            config = {
                "video": str(video_path.name),
                "frame_width": w,
                "frame_height": h,
                "setup_frame": args.frame,
                "exit_lines": {
                    line_data["label"]: line_data["line"]
                    for line_data in saved_lines
                }
            }

            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"\n  ✓ Config saved to: {output_path}")
            print(f"  Lines defined: {[l['label'] for l in saved_lines]}")
            print(f"\n  Next step: load this in your main pipeline with:")
            print(f"    python main_tracking.py --video {args.video} --config {output_path}\n")
            break

        elif key in (ord('q'), 27):
            print("\n  Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()