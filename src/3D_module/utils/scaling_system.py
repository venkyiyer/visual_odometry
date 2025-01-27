import cv2
import numpy as np

def extract_features_orb(image):
    """
    Extract ORB features from an image.
    :param image: input image
    :return: keypoints and descriptors
    """
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def extract_features_sift(image):
    """
    Extract SIFT features from an image.
    :param image: input image
    :return: keypoints and descriptors
    """
    if hasattr(cv2, 'SIFT'):
        sift = cv2.SIFT_create()
    else:
        sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features_orb(descriptors1, descriptors2):
    """
    Match ORB features between two sets of descriptors.
    :param descriptors1: descriptors from the first image
    :param descriptors2: descriptors from the second image
    :return: sorted matches
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return sorted(matches, key=lambda x: x.distance)

def match_features_sift(descriptors1, descriptors2):
    """
    Match SIFT features between two sets of descriptors.
    :param descriptors1: descriptors from the first image
    :param descriptors2: descriptors from the second image
    :return: sorted matches
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return sorted(matches, key=lambda x: x.distance)

def associate_depth(keypoints1, keypoints2, matches, depth_image):
    """
    Associate depth values with matched keypoints.
    :param keypoints1: keypoints from the first image
    :param keypoints2: keypoints from the second image
    :param matches: matched keypoints
    :param depth_image: depth image
    :return: depth associations
    """
    depth_associations = []
    for match in matches:
        if match.queryIdx >= len(keypoints1) or match.trainIdx >= len(keypoints2):
            continue
        x1, y1 = keypoints1[match.queryIdx].pt
        if not (0 <= int(x1) < depth_image.shape[1] and 0 <= int(y1) < depth_image.shape[0]):
            continue
        depth1 = depth_image[int(y1), int(x1)]
        if depth1 != 0:
            depth_associations.append((match, depth1))
    return depth_associations

def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    """
    Convert pixel coordinates to 3D camera-centric coordinates.
    :param u: x-coordinate in the image
    :param v: y-coordinate in the image
    :param depth: depth value
    :param fx: focal length x
    :param fy: focal length y
    :param cx: principal point x
    :param cy: principal point y
    :return: 3D coordinates
    """
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return np.array([x, y, z])

def calculate_displacements(keypoints1, keypoints2, depth_associations1, depth_associations2, fx, fy, cx, cy):
    """
    Calculate displacements between matched keypoints.
    :param keypoints1: keypoints from the first image
    :param keypoints2: keypoints from the second image
    :param depth_associations1: depth associations for the first image
    :param depth_associations2: depth associations for the second image
    :param fx: focal length x
    :param fy: focal length y
    :param cx: principal point x
    :param cy: principal point y
    :return: displacements
    """
    displacements = []
    for (match1, depth1), (match2, depth2) in zip(depth_associations1, depth_associations2):
        u1, v1 = keypoints1[match1.queryIdx].pt
        u2, v2 = keypoints2[match2.trainIdx].pt
        pos1 = pixel_to_3d(u1, v1, depth1, fx, fy, cx, cy)
        pos2 = pixel_to_3d(u2, v2, depth2, fx, fy, cx, cy)
        displacements.append(pos2 - pos1)
    return displacements

def compute_scaling_factor(curr_rgb, prev_rgb, curr_dp, prev_dp, intrinsics, feature_type="orb"):
    """
    Compute the scaling factor between two frames.
    :param curr_rgb: current RGB image
    :param prev_rgb: previous RGB image
    :param curr_dp: current depth image
    :param prev_dp: previous depth image
    :param intrinsics: camera intrinsics
    :param feature_type: feature extraction method ("orb" or "sift")
    :return: average displacement
    """
    fx, fy, cx, cy = intrinsics
    if feature_type == "sift":
        keypoints1, descriptors1 = extract_features_sift(prev_rgb)
        keypoints2, descriptors2 = extract_features_sift(curr_rgb)
        matches = match_features_sift(descriptors1, descriptors2)
    else:
        keypoints1, descriptors1 = extract_features_orb(prev_rgb)
        keypoints2, descriptors2 = extract_features_orb(curr_rgb)
        matches = match_features_orb(descriptors1, descriptors2)
    depth_associations_prev = associate_depth(keypoints1, keypoints2, matches, prev_dp)
    depth_associations_curr = associate_depth(keypoints2, keypoints1, matches, curr_dp)
    displacements = calculate_displacements(keypoints1, keypoints2, depth_associations_prev, depth_associations_curr, fx, fy, cx, cy)
    return np.mean(displacements, axis=0)


