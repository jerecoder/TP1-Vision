import matplotlib.pyplot as plt
import cv2
from utils import imshow, show_images
import numpy as np
from scipy.spatial import cKDTree

def sift_images(images):
    sift_algo = cv2.SIFT_create()
    images_keypoints = []
    images_descriptors = []

    for i in range(len(images)):
        kp, des = sift_algo.detectAndCompute(images[i], None)
        images_keypoints.append(kp)
        images_descriptors.append(des)

    return images_keypoints, images_descriptors

def adaptive_non_max_suppression(keypoints, descriptors, base_radius=10.0, max_kp=10000):
    if not keypoints:
        return [], None

    pts = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])
    sizes = np.array([kp.size for kp in keypoints])

    order = np.argsort(-responses)
    tree = cKDTree(pts)
    suppressed = np.zeros(len(keypoints), dtype=bool)
    keep = []

    for i in order:
        if suppressed[i]:
            continue

        keep.append(i)
        if len(keep) >= max_kp:
            break

        adaptive_radius = base_radius * (sizes[i] / np.mean(sizes))
        neighbors = tree.query_ball_point(pts[i], adaptive_radius)
        suppressed[neighbors] = True
        suppressed[i] = False 

    filtered_keypoints = [keypoints[i] for i in keep]
    filtered_descriptors = descriptors[keep] if descriptors is not None else None

    return filtered_keypoints, filtered_descriptors

def apply_anms(keypoints, descriptors):
    for i in range(len(keypoints)):
        keypoints[i], descriptors[i] = adaptive_non_max_suppression(
            keypoints[i],
            descriptors[i],
            max_kp=10000
        )

    return keypoints, descriptors

def bf_lowe_test(desc1, desc2, ratio=0.6):
    bf = cv2.BFMatcher(cv2.NORM_L2)

    desc1 = desc1.astype(np.float32)
    desc2 = desc2.astype(np.float32)
    
    knn_matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

def dlt(ori, dst):

    # Construct matrix A and vector b
    A = []
    b = []
    for i in range(4):
        x, y = ori[i]
        x_prima, y_prima = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prima, y * x_prima])
        A.append([0, 0, 0, -x, -y, -1, x * y_prima, y * y_prima])
        b.append(x_prima)
        b.append(y_prima)

    A = np.array(A)
    b = np.array(b)

    # resolvemos el sistema de ecuaciones A * h = b
    # el sistema es de 8x8, por lo que podemos resolverlo si A es inversible

    # resuelve el sistema de ecuaciones para encontrar los parámetros de H
    H = -np.linalg.solve(A, b)

    # agrega el elemento h_33
    H = np.hstack([H, [1]])

    # reorganiza H para formar la matrix en 3x3 to form the 3x3 homography matrix
    H = H.reshape(3, 3)

    return H

def ransac_homography_from_matches(kp1: list, kp2: list, matches: list, max_iterations: int = 1000, threshold: float = 5.0):
    max_inliers = 0
    best_inliers = []
    best_H = None

    for _ in range(max_iterations):
        sample_matches = np.random.choice(matches, 4, replace=False)

        # Get the point correspondences
        points1 = [kp1[m.queryIdx].pt for m in sample_matches]
        points2 = [kp2[m.trainIdx].pt for m in sample_matches]

        # Estimate homography using the improved DLT
        H = dlt(points1, points2)

        inliers = []
        for m in matches:
            pt1 = np.array([*kp1[m.queryIdx].pt, 1], dtype=np.float32)
            pt2 = np.array([*kp2[m.trainIdx].pt, 1], dtype=np.float32)

            projected_pt = H @ pt1
            projected_pt /= projected_pt[2]  # Normalize homogeneous coordinates
            # Calculate distance in image plane (x, y)
            distance = np.linalg.norm(projected_pt[:2] - pt2[:2])

            if distance < threshold:
                inliers.append(m)

        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_inliers = inliers
            best_H = H

    return best_H, best_inliers

def cylindrical_warp(img, f):
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    xs_c = xs - w / 2
    ys_c = ys - h / 2
    theta = np.arctan(xs_c / f)
    h_ = ys_c / np.sqrt(xs_c**2 + f**2)
    map_x = (f * theta + w / 2).astype(np.float32)
    map_y = (f * h_ + h / 2).astype(np.float32)
    cyl = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return cyl

def compute_panorama_bounds(images, H_matrices):
    all_corners = []
    for img, H in zip(images, H_matrices):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped_corners)
    all_corners = np.vstack(all_corners)
    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))
    return x_min, y_min, x_max, y_max

def create_panoramic(images):
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    keypoints, descriptors = sift_images(gray_images)
    keypoints, descriptors = apply_anms(keypoints, descriptors)
    matches = [None, None]
    
    matches[0] = bf_lowe_test(descriptors[0], descriptors[1])
    matches[1] = bf_lowe_test(descriptors[1], descriptors[0])

    inliers = [None, None]

    _, inliers[0] = ransac_homography_from_matches(keypoints[0], keypoints[1], matches[0])
    _, inliers[1] = ransac_homography_from_matches(keypoints[1], keypoints[0], matches[1])

    points_0_src = np.array([keypoints[0][m.queryIdx].pt for m in inliers[0]])
    points_0_dst = np.array([keypoints[1][m.trainIdx].pt for m in inliers[0]])

    points_2_src = np.array([keypoints[1][m.queryIdx].pt for m in inliers[1]])
    points_2_dst = np.array([keypoints[0][m.trainIdx].pt for m in inliers[1]])

    H_0, _ = cv2.findHomography(points_0_src, points_0_dst)
    H1 = np.eye(3)
    H_2_inv, _ = cv2.findHomography(points_2_src, points_2_dst)

    H_2 = np.linalg.inv(H_2_inv)

    H_matrices = [H_0, H1, H_2]
    
    focal_length = 5000
    images_cyl = [cylindrical_warp(img, focal_length) for img in images]
    x_min, y_min, x_max, y_max = compute_panorama_bounds(images_cyl, H_matrices)

    translation = np.array([[1, 0, -x_min],
                        [0, 1, -y_min],
                        [0, 0, 1]])
    
    pano_width = x_max - x_min
    pano_height = y_max - y_min

    warped_images = []
    for idx, (img, H) in enumerate(zip(images_cyl, H_matrices)):
        warped = cv2.warpPerspective(img, translation @ H, (pano_width, pano_height))
        warped_images.append(warped)

    accum = np.zeros((pano_height, pano_width, 3), dtype=np.float32)
    weight_accum = np.zeros((pano_height, pano_width), dtype=np.float32)

    for warped in warped_images:
        mask = np.any(warped > 0, axis=2).astype(np.uint8)
        mask_8u = (mask * 255).astype(np.uint8)
        dist = cv2.distanceTransform(mask_8u, cv2.DIST_L2, 3)
        weight = dist[..., np.newaxis]
        
        accum += warped.astype(np.float32) * weight
        weight_accum += dist

    weight_accum_3c = np.repeat(weight_accum[..., np.newaxis], 3, axis=2)
    panorama = accum / np.maximum(weight_accum_3c, 1e-6)
    panorama = panorama.astype(np.uint8)

    return panorama

def main():
    images = [cv2.imread(f'img/udesa_{i}.jpg') for i in range(3)]
    panorama = create_panoramic(images)
    imshow(panorama)

if __name__ == "__main__":
    main()



