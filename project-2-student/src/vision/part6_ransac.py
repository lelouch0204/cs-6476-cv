import math

import numpy as np
import cv2


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: int
) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float representing the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    if ind_prob_correct == 1:
        return 1
    
    num_samples = math.log(1 - prob_success) / math.log(1 - ind_prob_correct**sample_size)
    num_samples = math.ceil(num_samples)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)

def ransac_homography(
    points_a: np.ndarray, points_b: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Uses the RANSAC algorithm to robustly estimate a homography matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) of points from image A.
    -   points_b: A numpy array of shape (N, 2) of corresponding points from image B.

    Returns:
    -   best_H: The best homography matrix of shape (3, 3).
    -   inliers_a: The subset of points_a that are inliers (M, 2).
    -   inliers_b: The subset of points_b that are inliers (M, 2).
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    #                                                                         #
    # HINT: You are allowed to use the `cv2.findHomography` function to       #
    # compute the homography from a sample of points. To compute a direct     #
    # solution without OpenCV's built-in RANSAC, use it like this:            #
    #   H, _ = cv2.findHomography(sample_a, sample_b, 0)                      #
    # The `0` flag ensures it computes a direct least-squares solution.       #
    ###########################################################################

    inlier_prob = 0.1
    sample_size = 4
    prob_success = 0.99
    threshold = 5.0

    num_iter = calculate_num_ransac_iterations(prob_success=prob_success, sample_size=sample_size, ind_prob_correct=inlier_prob)
    N = points_a.shape[0]

    best_H = None
    best_inliers = []

    for _ in range(num_iter):
        indices = np.random.choice(N, size=4, replace=False)

        sample_a = points_a[indices]
        sample_b = points_b[indices]

        H, _ = cv2.findHomography(sample_a, sample_b, 0)

        if H is None:
            continue

        points_a_h = np.hstack([points_a, np.ones((N, 1))])
        proj_b = (H @ points_a_h.T).T

        proj_b = proj_b[:, :2]/proj_b[:, [2]]

        errors = np.linalg.norm(points_b - proj_b, axis=1)
        inliers = np.where(errors < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    if len(best_inliers) > 4:
        best_H, _ = cv2.findHomography(points_a[best_inliers], points_b[best_inliers], 0)
    
    inliers_a = points_a[best_inliers]
    inliers_b = points_b[best_inliers]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_H, inliers_a, inliers_b
