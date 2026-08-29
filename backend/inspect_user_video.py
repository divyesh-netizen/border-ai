import cv2
import os
import sys

video_path = r"d:\SIH\WhatsApp Video 2026-08-29 at 3.38.08 PM.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open {video_path}")
    sys.exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / max(1, fps)

print(f"Video Info:")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"Duration: {duration:.2f}s")

# Copy to uploads so it's also accessible from web UI
uploads_dir = r"d:\SIH\uploads"
os.makedirs(uploads_dir, exist_ok=True)
dest_path = os.path.join(uploads_dir, "whatsapp_surveillance.mp4")
import shutil
shutil.copyfile(video_path, dest_path)
print(f"Copied to uploads as: {dest_path}")

# Extract sample frames to inspect
output_debug_dir = r"d:\SIH\outputs\debug_frames"
os.makedirs(output_debug_dir, exist_ok=True)

sample_indices = [int(total_frames * 0.1), int(total_frames * 0.3), int(total_frames * 0.5), int(total_frames * 0.7), int(total_frames * 0.9)]
for idx in sample_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(output_debug_dir, f"frame_{idx}.jpg"), frame)
        print(f"Saved debug frame {idx}")

cap.release()
