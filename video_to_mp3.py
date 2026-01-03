# converts videos to mp3
# Converts all videos in "videos/" folder to MP3 and saves in "audios/"
import os
import re
import subprocess

input_dir = "videos"
output_dir = "audios"


os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if not file.lower().endswith(".mp4"):
        print(f"Skipping non-video file: {file}")
        continue

    # Extract tutorial number (last number before .mp4)
    match = re.search(r'(\d+)(?=\.mp4$)', file)
    tutorial_number = match.group(1) if match else "unknown"

    # Remove .mp4 and sanitize name
    file_name = os.path.splitext(file)[0]
    safe_file_name = re.sub(r'[\\/*?:"<>|]', "", file_name).strip()

    # Build input and output paths
    input_path = os.path.join(input_dir, file)
    output_file = f"{tutorial_number}_{safe_file_name}.mp3"
    output_path = os.path.join(output_dir, output_file)

    print(f"🎧 Converting: {file} → {output_file}")

    # Run ffmpeg
    subprocess.run([
        "ffmpeg",
        "-y",  # overwrite if exists
        "-i", input_path,
        "-q:a", "0",  # high quality audio
        "-map", "a",  # select audio stream
        output_path
    ])
