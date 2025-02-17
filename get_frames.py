import cv2
import os

def extract_frames(video_path, output_folder, interval=3):
    video_cap = cv2.VideoCapture(video_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps) * interval

    os.mkdir(output_folder)

    frame_count = 0
    saved_count = 0

    while True:
        success, frame = video_cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            output_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1


    video_cap.release()

    return saved_count


video1_path = './hw4_1.mp4'
video2_path = './hw4_2.mp4'

output_folder1 = './frames_video1'
output_folder2 = './frames_video2'

# frames_video1 = extract_frames(video1_path, output_folder1, interval=3)
frames_video2 = extract_frames(video2_path, output_folder2, interval=1)

