import os

files = os.listdir("videos")

print("\nFiles inside 'videos' folder:\n")
for f in files:
    print(f"- {f}")
