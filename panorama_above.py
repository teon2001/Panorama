import cv2
import os

cv2.ocl.setUseOpenCL(False)

def create_panorama(frame_folder, output_path):
    frame_paths = [os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg')]
    frames = [cv2.imread(f) for f in frame_paths]

    stitcher = cv2.Stitcher_create() if cv2.__version__.startswith('4') else cv2.createStitcher()
    status, panorama = stitcher.stitch(frames)

    if status == cv2.Stitcher_OK:
        cv2.imwrite(output_path, panorama)
        print(f"Panorama saved at {output_path}")
        return True
    else:
        print(f"Stitching failed with status code {status}")
        return False



frame_folder = "./frames_video1"
output_path = "./panorama_above.jpg"

create_panorama(frame_folder, output_path)