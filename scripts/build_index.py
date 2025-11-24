import csv
from pathlib import Path
import pandas as pd

# Root project directory (change if your path is different)
ROOT = Path(r"D:\GU\paper\surgclip")

FRAMES_DIR = ROOT / "frames"
PHASE_DIR = ROOT / "data" / "cholec80" / "phase_annotations"
INDEX_CSV = ROOT / "index" / "keyframe_index.csv"

# Natural language descriptions for each phase
def phase_name_to_id_and_text(phase_name: str):
    """
    Map a raw phase name from the annotation file (e.g. 'Preparation')
    to a numeric id in [1..7] and a more descriptive English sentence.
    """
    name = str(phase_name).lower()

    if "prep" in name:
        return 1, (
            "The surgical tools are being inserted and the field is prepared "
            "for the procedure."
        )
    elif "calot" in name:
        return 2, (
            "The surgeon is dissecting tissue in Calot's triangle to expose "
            "the cystic duct and artery."
        )
    elif "clip" in name or "cut" in name:
        return 3, (
            "The cystic duct and artery are being clipped and cut to "
            "separate them safely."
        )
    elif "dissection" in name and "gallbladder" in name:
        return 4, (
            "The gallbladder is being dissected away from the liver bed."
        )
    elif "pack" in name or "bag" in name:
        return 5, (
            "The gallbladder is being packed and prepared for removal."
        )
    elif "clean" in name or "coag" in name:
        return 6, (
            "Residual bleeding or bile leakage is being controlled by "
            "cleaning and coagulation."
        )
    elif "retrac" in name or "retract" in name:
        return 7, (
            "The gallbladder is being extracted from the abdominal cavity."
        )
    else:
        # Fallback: unknown / unexpected label
        return 0, f"In phase '{phase_name}' of the laparoscopic procedure."


def build_phase_file_map():
    """
    Build a mapping from video_id (e.g. 'video01') to its phase annotation file.
    We assume phase annotation files contain 'phase' in their name
    (e.g. 'video01-phase.txt').
    """
    mapping = {}

    for path in PHASE_DIR.glob("*.txt"):
        name = path.name.lower()
        if "phase" not in name:
            # Skip tool annotation files (they usually contain 'tool' instead)
            continue

        # Typical file name: "video01-phase.txt" -> key "video01"
        key = name.split("-")[0]  # part before first '-'
        mapping[key] = path

    return mapping


def load_phase_table(phase_file: Path):
    """
    Load the phase annotation table.
    Each file has two columns: frame_index, phase_label (as a string).
    We allow whitespace or comma separated formats.
    """
    try:
        df = pd.read_csv(phase_file, sep=r"\s+|,", engine="python", header=0)
    except Exception as e:
        print(f"[ERROR] Failed to read phase file {phase_file}: {e}")
        return None

    # Keep only the first two columns: frame index and phase label
    df = df.iloc[:, :2]
    df.columns = ["frame_idx", "phase_name"]

    # Sort by frame index just in case
    df = df.sort_values("frame_idx").reset_index(drop=True)
    return df


def get_phase_for_frame(df, orig_frame_idx):
    """
    Given a phase table (df) and an estimated original frame index,
    return the corresponding phase_name (string).

    Strategy:
    - Find the last row where frame_idx <= orig_frame_idx.
    - If none is found (e.g. very early frame), use the first row.
    """
    mask = df["frame_idx"] <= orig_frame_idx
    if mask.any():
        idx = mask[mask].index[-1]
        return str(df.loc[idx, "phase_name"])
    else:
        # Fallback: use the first phase name
        return str(df.loc[0, "phase_name"])


def main():
    # Original Cholec80 videos are recorded at 25 fps (according to README)
    # We extracted frames at 1 fps, so each saved frame roughly corresponds
    # to every 25th original frame.
    EST_FPS = 25

    # Build mapping from video id to phase annotation file
    vid_to_phase = build_phase_file_map()
    if not vid_to_phase:
        print(f"[ERROR] No phase annotation files found in {PHASE_DIR}")
        return

    print("[INFO] Found phase annotation files for:")
    for k in sorted(vid_to_phase.keys()):
        print(f"  - {k}")

    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)

    with INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame_path", "video_id", "phase_id", "phase_name", "phase_text"]
        )

        # Iterate over each video folder in frames/
        for video_dir in sorted(FRAMES_DIR.iterdir()):
            if not video_dir.is_dir():
                continue

            video_id = video_dir.name  # e.g. "video01"
            key = video_id.lower()

            if key not in vid_to_phase:
                print(f"[WARN] No phase file found for {video_id}, skipping.")
                continue

            phase_file = vid_to_phase[key]
            df_phase = load_phase_table(phase_file)
            if df_phase is None:
                print(f"[WARN] Failed to load phase file for {video_id}, skipping.")
                continue

            print(f"[INFO] Processing {video_id} with phase file {phase_file.name}")

            frame_files = sorted(video_dir.glob("frame_*.jpg"))
            for frame_path in frame_files:
                # Extract saved frame index from file name: frame_000123.jpg -> 123
                stem = frame_path.stem  # "frame_000123"
                try:
                    saved_idx = int(stem.split("_")[1])
                except (IndexError, ValueError):
                    print(f"[WARN] Unexpected frame name: {frame_path.name}, skipping.")
                    continue

                # Estimate original frame index in 25-fps video
                orig_frame_idx = saved_idx * EST_FPS

                # Look up phase name from annotation table
                phase_name = get_phase_for_frame(df_phase, orig_frame_idx)

                # Convert to (id, natural language text)
                phase_id, phase_text = phase_name_to_id_and_text(phase_name)

                # Write row with relative frame path
                writer.writerow(
                    [
                        str(frame_path.relative_to(ROOT)),
                        video_id,
                        phase_id,
                        phase_name,
                        phase_text,
                    ]
                )

    print(f"[INFO] Index CSV written to: {INDEX_CSV}")


if __name__ == "__main__":
    main()
