import os

import cv2
import numpy as np

def align_image_sizes(img1, img2):
    # Determinăm dimensiunile maxime
    height = max(img1.shape[0], img2.shape[0])
    width = max(img1.shape[1], img2.shape[1])

    # Extindem img1
    extended_img1 = np.zeros((height, width, 3), dtype=np.uint8)
    extended_img1[:img1.shape[0], :img1.shape[1]] = img1

    # Extindem img2
    extended_img2 = np.zeros((height, width, 3), dtype=np.uint8)
    extended_img2[:img2.shape[0], :img2.shape[1]] = img2

    return extended_img1, extended_img2


def stitch_images_sift(images):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Aliniem manual imaginile una câte una
    result = images[0]

    for i in range(1, len(images)):
        img1 = result
        img2 = images[i]

        # Detectăm caracteristicile și extragem descrierile
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Potrivim caracteristicile folosind FLANN-based matcher
        # Folosim un algoritm de căutare rapidă (FLANN) pentru a găsi
        # cele mai bune potriviri între descrierile celor două imagini.
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Aplicăm filtrul Lowe pentru potriviri bune
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Dacă există suficiente potriviri bune, calculăm omografia
        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculăm matricea omografică
            # Calculează o matrice 3×3, numită matricea omografică,
            # care descrie transformarea perspectivei între punctele sursă
            # (dst_pts) și punctele destinație (src_pts).
            # 5.0: Pragul de eroare pentru a decide dacă un punct este outlier sau nu
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            # Aplicăm transformarea perspectivei asupra imaginii curente
            h, w = img1.shape[:2]
            img2_aligned = cv2.warpPerspective(img2, M, (w, h + img2.shape[0]))

            # Extindem dimensiunile pentru a fi compatibile
            img1_resized, img2_resized = align_image_sizes(img1, img2_aligned)

            # Combinăm imaginile prin suprapunere
            result = np.maximum(img1_resized, img2_resized)
        else:
            print(f"Insufficient matches between images {i-1} and {i}")
            break

    return result

# Citim imaginile din folder
def load_images_from_folder(folder):
    image_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')])
    return [cv2.imread(img_path) for img_path in image_paths[:7]]

# Exemplu
frame_folder = "./frames_video2"
images = load_images_from_folder(frame_folder)

# Creăm panorama folosind SIFT
panorama = stitch_images_sift(images)

# Salvăm rezultatul
output_path = "./panorama_angle.jpg"
cv2.imwrite(output_path, panorama)
print(f"Panorama saved at {output_path}")
