import cv2
import os
from pathlib import Path
from tqdm import tqdm

# Root directory (modify to your project root path as needed)
ROOT = Path(r"D:\GU\paper\surgclip")
VIDEO_DIR = ROOT / "data" / "cholec80" / "videos"
FRAMES_DIR = ROOT / "frames"

FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# How many frames per second
TARGET_FPS = 1.0

def extract_frames_from_video(video_path: Path):
    """ Extract frames from a video by TARGET_FPS and save them to frames/<video_name>/. """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    # Get the original video frame rate
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 25.0  # fallback

    # How many frames should we extract once
    frame_interval = max(int(round(orig_fps / TARGET_FPS)), 1)

    video_id = video_path.stem  # Remove the file extension as the folder name
    out_dir = FRAMES_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            fname = out_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[INFO] {video_id}: saved {saved_idx} frames.")

def main():
    video_files = []
    for ext in ("*.mp4", "*.avi", "*.mkv"):
        video_files.extend(VIDEO_DIR.glob(ext))

    if not video_files:
        print(f"[ERROR] No video files found in {VIDEO_DIR}")
        return

    print(f"[INFO] Found {len(video_files)} videos.")
    for vp in tqdm(sorted(video_files), desc="Extracting frames"):
        extract_frames_from_video(vp)

if __name__ == "__main__":
    main()
