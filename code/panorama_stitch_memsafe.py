"""
PANORAMIC IMAGE STITCHING — FROM SCRATCH IMPLEMENTATION

Permitted library usage 
  - OpenCV: Image I/O, SIFT detection, keypoint matching
  - NumPy:  Matrix operations, eigendecomposition, linear algebra
  - Matplotlib: Visualization

Manual implementations (NO high-level APIs):
  - Homography estimation via Direct Linear Transform (DLT)
  - RANSAC for robust estimation
  - Perspective warping (forward + inverse mapping)
  - Multi-band blending (Laplacian pyramid) & feathered blending
  - Intensity normalization and final cleanup

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')



# STEP 1: IMAGE LOADING & PREPROCESSING


def load_images(image_paths):
    """
    Load a sequence of images from disk. We expect them ordered
    left-to-right (or the user can specify ordering). Each image
    is read in BGR (OpenCV default) and also converted to RGB
    for visualization, plus a grayscale copy for feature detection.
    """
    images_bgr = []
    images_gray = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Cannot read image at: {p}")
        images_bgr.append(img)
        images_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        print(f"  Loaded: {p}  ->  shape {img.shape}")
    return images_bgr, images_gray


def display_images(images_bgr, title="Input Images"):
    """Quick side-by-side display of the loaded images."""
    n = len(images_bgr)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for i, img in enumerate(images_bgr):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Image {i}")
        axes[i].axis('off')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("01_input_images.png", dpi=150, bbox_inches='tight')
    plt.close()



# STEP 2: SIFT FEATURE EXTRACTION  (OpenCV built-in permitted)


def extract_sift_features(images_gray):
    """
    Detect SIFT keypoints and compute descriptors for each grayscale image.

    SIFT (Scale-Invariant Feature Transform) identifies interest points that
    are invariant to scale and rotation. Each keypoint receives a 128-dim
    descriptor encoding the local gradient histogram around it.

    Returns:
        keypoints_list:  list of keypoint arrays per image
        descriptors_list: list of descriptor matrices (N x 128) per image
    """
    sift = cv2.SIFT_create(nfeatures=10000)  # generous upper bound

    keypoints_list = []
    descriptors_list = []

    for idx, gray in enumerate(images_gray):
        kp, des = sift.detectAndCompute(gray, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)
        print(f"  Image {idx}: detected {len(kp)} SIFT keypoints")

    return keypoints_list, descriptors_list


def visualize_keypoints(images_bgr, keypoints_list):
    """Draw detected keypoints on each image """
    n = len(images_bgr)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for i in range(n):
        vis = cv2.drawKeypoints(
            images_bgr[i], keypoints_list[i], None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        axes[i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Image {i} — {len(keypoints_list[i])} keypoints")
        axes[i].axis('off')
    fig.suptitle("SIFT Keypoints", fontsize=14)
    plt.tight_layout()
    plt.savefig("02_sift_keypoints.png", dpi=150, bbox_inches='tight')
    plt.close()



# STEP 3: FEATURE MATCHING  (OpenCV BFMatcher permitted)


def match_features(des1, des2, ratio_thresh=0.75):
    """
    Match descriptors between two images using Brute-Force with
    Lowe's ratio test.

    For each descriptor in image 1, we find its two nearest neighbors
    in image 2. If the distance to the closest match is significantly
    smaller than to the second closest (ratio < threshold), we accept
    the match. This filters out ambiguous correspondences.

    Args:
        des1, des2:     descriptor matrices (N1x128, N2x128)
        ratio_thresh:   Lowe's ratio test threshold

    Returns:
        good_matches:   list of cv2.DMatch objects passing the ratio test
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    print(f"    Ratio test: {len(raw_matches)} raw -> {len(good_matches)} good matches")
    return good_matches


def get_matched_points(kp1, kp2, matches):
    """
    Extract the (x, y) coordinates of matched keypoints into
    two corresponding Nx2 arrays.
    """
    pts1 = np.float64([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def visualize_matches(img1, kp1, img2, kp2, matches, title="Feature Matches", fname="matches.png"):
    """Draw matches between two images side by side."""
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:80], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(16, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()



# STEP 4: HOMOGRAPHY ESTIMATION - DIRECT LINEAR TRANSFORM (DLT) - MANUAL IMPLEMENTATION


def normalize_points(pts):
    """
    Hartley normalization: translate points so centroid is at origin,
    then scale so average distance from origin is sqrt(2).

    This conditioning dramatically improves the numerical stability
    of the DLT. Without it, the system matrix can be very ill-conditioned
    because pixel coordinates are O(1000) while homogeneous coordinates
    mix 1s with 1000s.

    Args:
        pts: Nx2 array of 2D points

    Returns:
        pts_norm: Nx2 normalized points
        T:        3x3 normalization matrix such that pts_norm = T @ pts_homogeneous
    """
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    avg_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
    if avg_dist < 1e-10:
        avg_dist = 1e-10
    scale = np.sqrt(2.0) / avg_dist

    T = np.array([
        [scale,  0,     -scale * centroid[0]],
        [0,      scale, -scale * centroid[1]],
        [0,      0,      1                  ]
    ])

    pts_h = np.column_stack([pts, np.ones(len(pts))])  # Nx3
    pts_norm_h = (T @ pts_h.T).T                        # Nx3
    pts_norm = pts_norm_h[:, :2] / pts_norm_h[:, 2:3]

    return pts_norm, T


def compute_homography_dlt(src_pts, dst_pts):
    """
    Compute the 3x3 homography H that maps src_pts -> dst_pts
    using the Direct Linear Transform (DLT) algorithm.

    Given a correspondence  (x, y) <-> (x', y'), the projective
    relation is:
        [x']       [x]
        [y'] ~ H * [y]
        [1 ]       [1]

    Expanding and eliminating the scale factor, each correspondence
    yields two linear equations in the 9 entries of H. Stacking N>=4
    correspondences produces the system  A h = 0, where h = vec(H).
    The solution is the right singular vector of A corresponding to
    the smallest singular value (i.e., the null space of A).

    We apply Hartley normalization to both point sets before solving,
    then de-normalize the result.

    Args:
        src_pts: Nx2 source points
        dst_pts: Nx2 destination points  (N >= 4)

    Returns:
        H: 3x3 homography matrix
    """
    assert len(src_pts) >= 4, "Need at least 4 correspondences for homography"

    # --- Hartley normalization ---
    src_norm, T_src = normalize_points(src_pts)
    dst_norm, T_dst = normalize_points(dst_pts)

    N = len(src_norm)
    A = np.zeros((2 * N, 9))

    for i in range(N):
        x, y = src_norm[i]
        xp, yp = dst_norm[i]

        # Row 2i:    -x  -y  -1   0   0   0   xp*x  xp*y  xp
        A[2 * i] = [
            -x, -y, -1,
             0,  0,  0,
             xp * x, xp * y, xp
        ]
        # Row 2i+1:   0   0   0  -x  -y  -1   yp*x  yp*y  yp
        A[2 * i + 1] = [
             0,  0,  0,
            -x, -y, -1,
             yp * x, yp * y, yp
        ]

    # Solve via SVD: the solution h is the last row of Vt (smallest singular value)
    _, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_norm = h.reshape(3, 3)

    # --- De-normalize ---
    # If  T_dst * dst ~ H_norm * T_src * src,  then  dst ~ T_dst^{-1} H_norm T_src * src
    H = np.linalg.inv(T_dst) @ H_norm @ T_src

    # Normalize so that H[2,2] = 1 (conventional, avoids scale ambiguity)
    if abs(H[2, 2]) > 1e-10:
        H /= H[2, 2]

    return H



# STEP 5: RANSAC — ROBUST ESTIMATION OF HOMOGRAPHY


def compute_homography_ransac(src_pts, dst_pts,
                               n_iterations=10000,
                               reprojection_thresh=3.0,
                               min_inliers_ratio=0.1):
    """
    RANSAC wrapper around the DLT homography estimator.

    RANSAC (Random Sample Consensus) iteratively:
      1. Samples a minimal subset (4 correspondences for homography).
      2. Fits a model (homography via DLT) to that subset.
      3. Counts how many of the remaining correspondences agree
         with the model (inliers within reprojection_thresh pixels).
      4. Keeps the model with the largest inlier set.
    Finally, re-estimates H from ALL inliers of the best model.

    This makes the estimation robust to outliers produced by
    incorrect feature matches (which are common in practice).

    Args:
        src_pts, dst_pts:     Nx2 matched point arrays
        n_iterations:         number of RANSAC trials
        reprojection_thresh:  max pixel error to count as inlier
        min_inliers_ratio:    minimum fraction of inliers to accept

    Returns:
        best_H:       3x3 homography (refined on all inliers)
        inlier_mask:  boolean array, True for inlier correspondences
    """
    N = len(src_pts)
    if N < 4:
        raise ValueError(f"Need >=4 matches for homography, got {N}")

    best_inlier_count = 0
    best_inlier_mask = None
    best_H = None

    # Pre-compute homogeneous source points for reprojection
    src_h = np.column_stack([src_pts, np.ones(N)])  # Nx3

    for iteration in range(n_iterations):
        # 1) Random minimal sample of 4 points
        indices = np.random.choice(N, 4, replace=False)
        s = src_pts[indices]
        d = dst_pts[indices]

        # Skip degenerate configurations (collinear points)
        # A quick check: if 3 of 4 source points are nearly collinear, skip
        try:
            H_candidate = compute_homography_dlt(s, d)
        except np.linalg.LinAlgError:
            continue

        if H_candidate is None or np.any(np.isnan(H_candidate)):
            continue

        # 2) Reproject ALL source points using H_candidate
        projected_h = (H_candidate @ src_h.T).T  # Nx3
        # Avoid division by zero
        w = projected_h[:, 2:3]
        w[np.abs(w) < 1e-10] = 1e-10
        projected = projected_h[:, :2] / w

        # 3) Compute reprojection error
        errors = np.sqrt(np.sum((projected - dst_pts) ** 2, axis=1))
        inlier_mask = errors < reprojection_thresh
        inlier_count = np.sum(inlier_mask)

        # 4) Update best model
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask.copy()
            best_H = H_candidate.copy()

        # Early termination: if we have a really good model, stop
        if inlier_count > 0.9 * N:
            break

    # Validate that we found enough inliers
    if best_inlier_count < min_inliers_ratio * N:
        raise RuntimeError(
            f"RANSAC failed: only {best_inlier_count}/{N} inliers "
            f"({100*best_inlier_count/N:.1f}%), threshold was {min_inliers_ratio*100:.0f}%"
        )

    # 5) Refine H using ALL inliers
    best_H = compute_homography_dlt(
        src_pts[best_inlier_mask],
        dst_pts[best_inlier_mask]
    )

    print(f"    RANSAC: {best_inlier_count}/{N} inliers "
          f"({100*best_inlier_count/N:.1f}%) over {min(iteration+1, n_iterations)} iterations")

    return best_H, best_inlier_mask


# =====================================================================================
# STEP 6: IMAGE WARPING - inverse mapping with bilinear interp - MANUAL IMPLEMENTATION
# =====================================================================================

def apply_homography_to_point(H, pt):
    """Apply homography H to a single 2D point, returning the mapped 2D point."""
    p = np.array([pt[0], pt[1], 1.0])
    q = H @ p
    if abs(q[2]) < 1e-10:
        q[2] = 1e-10
    return q[0] / q[2], q[1] / q[2]


def compute_warped_bounds(H, h, w):
    """
    Determine the bounding box of an image of size (h, w) after
    warping by homography H by mapping its four corners.

    Returns (x_min, y_min, x_max, y_max) in the destination coordinate frame.
    """
    corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float64)

    warped_corners = []
    for c in corners:
        wx, wy = apply_homography_to_point(H, c)
        warped_corners.append([wx, wy])
    warped_corners = np.array(warped_corners)

    x_min = np.floor(warped_corners[:, 0].min()).astype(int)
    x_max = np.ceil(warped_corners[:, 0].max()).astype(int)
    y_min = np.floor(warped_corners[:, 1].min()).astype(int)
    y_max = np.ceil(warped_corners[:, 1].max()).astype(int)

    return x_min, y_min, x_max, y_max


def bilinear_interpolate(img, x, y):
    """
    Bilinear interpolation at sub-pixel location (x, y) on a BGR image.

    Instead of simply rounding to the nearest pixel (which causes
    aliasing), we compute a weighted average of the four surrounding
    pixels. The weights are proportional to the area of the opposite
    rectangle in the unit cell.

    Args:
        img:  HxWx3 image (uint8 or float)
        x, y: float coordinates (x = column, y = row)

    Returns:
        Interpolated pixel value (3-element array), or zeros if out of bounds.
    """
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    # Boundary check
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return np.zeros(channels, dtype=np.float64)

    # Fractional parts
    dx = x - x0
    dy = y - y0

    # Four neighbor values
    val = (
        (1 - dx) * (1 - dy) * img[y0, x0].astype(np.float64) +
        dx       * (1 - dy) * img[y0, x1].astype(np.float64) +
        (1 - dx) * dy       * img[y1, x0].astype(np.float64) +
        dx       * dy       * img[y1, x1].astype(np.float64)
    )
    return val


def warp_image(img, H, output_size, offset):
    """
    Memory-Safe Warping: Processes the image in chunks (scanlines) 
    to avoid crashing RAM on large panoramas.
    """
    out_h, out_w = output_size
    H_inv = np.linalg.inv(H)
    off_x, off_y = offset
    
    # 1. Allocate the final destination arrays
    #    If this line specifically fails, you simply don't have enough RAM 
    #    to hold the result, and we'd need to use 'np.memmap' (disk-based array).
    try:
        warped = np.zeros((out_h, out_w, 3), dtype=np.float64)
        mask = np.zeros((out_h, out_w), dtype=np.float64)
    except MemoryError:
        print("\n[!] Critical: System RAM cannot hold the final image.")
        print("    Switching to disk-based memory mapping (slower, but works).")
        warped = np.memmap('temp_warped.dat', dtype='float64', mode='w+', shape=(out_h, out_w, 3))
        mask = np.memmap('temp_mask.dat', dtype='float64', mode='w+', shape=(out_h, out_w))

    h_src, w_src = img.shape[:2]

    # 2. Process in Chunks (e.g., 500 rows at a time)
    #    This ensures intermediate math arrays never explode RAM.
    CHUNK_SIZE = 500  
    
    print(f"  -> Processing {out_h} rows in chunks of {CHUNK_SIZE}...")

    for y_start in range(0, out_h, CHUNK_SIZE):
        y_end = min(y_start + CHUNK_SIZE, out_h)
        chunk_h = y_end - y_start
        
        # --- Generate Grid for THIS CHUNK ONLY ---
        # Create grid of output pixel coordinates for the current strip
        ys, xs = np.mgrid[y_start:y_end, 0:out_w]
        dest_x = xs + off_x
        dest_y = ys + off_y

        # Flatten for matrix multiplication
        ones = np.ones(chunk_h * out_w, dtype=np.float64)
        flat_x = dest_x.reshape(-1)
        flat_y = dest_y.reshape(-1)
        
        # Stack: (N, 3) -> [x, y, 1]
        dest_coords = np.vstack([flat_x, flat_y, ones]) # 3 x N

        # --- Apply Inverse Homography ---
        # src = H_inv @ dest
        src_coords = H_inv @ dest_coords
        
        # Perspective divide
        w_vals = src_coords[2, :]
        w_vals[np.abs(w_vals) < 1e-10] = 1e-10
        src_x = src_coords[0, :] / w_vals
        src_y = src_coords[1, :] / w_vals

        # Reshape back to chunk shape
        src_x = src_x.reshape(chunk_h, out_w)
        src_y = src_y.reshape(chunk_h, out_w)

        # --- Vectorized Bilinear Interpolation (Chunk Scope) ---
        
        # 1. Determine valid pixels
        valid = (src_x >= 0) & (src_x < w_src - 1) & \
                (src_y >= 0) & (src_y < h_src - 1)
        
        # If no valid pixels in this chunk, skip math
        if not np.any(valid):
            continue

        # 2. Get integer and fractional parts
        # We only compute for 'valid' pixels to save more speed/memory, 
        # but masking arrays is cleaner in numpy.
        x0 = np.floor(src_x).astype(int)
        y0 = np.floor(src_y).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        
        dx = src_x - x0
        dy = src_y - y0

        # Clamp to bounds to prevent index errors on edges
        x0 = np.clip(x0, 0, w_src - 2)
        x1 = np.clip(x1, 0, w_src - 1)
        y0 = np.clip(y0, 0, h_src - 2)
        y1 = np.clip(y1, 0, h_src - 1)

        # 3. Fetch neighbors
        Ia = img[y0, x0].astype(np.float64)
        Ib = img[y0, x1].astype(np.float64)
        Ic = img[y1, x0].astype(np.float64)
        Id = img[y1, x1].astype(np.float64)

        # 4. Compute weights (re-expand dimensions for broadcasting)
        wa = ((1 - dx) * (1 - dy))[:, :, None]
        wb = (dx * (1 - dy))[:, :, None]
        wc = ((1 - dx) * dy)[:, :, None]
        wd = (dx * dy)[:, :, None]

        # 5. Interpolate
        # This line was crashing before. Now it's 1/100th the size.
        chunk_warped = wa * Ia + wb * Ib + wc * Ic + wd * Id

        # 6. Assign to final array
        # We apply the validity mask here
        # Expand valid to 3 channels for image, keep 1 channel for mask
        valid_3ch = valid[:, :, None]
        
        # Copy strictly the valid region into the main array
        # (Using 'np.where' is safer than boolean indexing for assignment)
        current_slice_img = warped[y_start:y_end, :]
        current_slice_mask = mask[y_start:y_end, :]
        
        np.copyto(current_slice_img, chunk_warped, where=valid_3ch)
        np.copyto(current_slice_mask, 1.0, where=valid)

    return warped, mask



# STEP 7: BLENDING — MULTI-BAND LAPLACIAN PYRAMID BLENDING & FEATHERING


def gaussian_blur_manual(img, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur. We build the kernel manually, but use
    cv2.filter2D for the actual convolution (which is a basic
    matrix operation, not a high-level stitching API).
    """
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def build_gaussian_pyramid(img, levels):
    """Construct a Gaussian pyramid by repeated blur + downsample."""
    pyramid = [img.astype(np.float64)]
    current = img.astype(np.float64)
    for _ in range(levels - 1):
        blurred = gaussian_blur_manual(current, kernel_size=5, sigma=1.0)
        # Downsample by factor of 2
        downsampled = blurred[::2, ::2]
        pyramid.append(downsampled)
        current = downsampled
    return pyramid


def upsample(img, target_shape):
    """Upsample an image to target_shape using simple pixel duplication + blur."""
    h, w = target_shape[:2]
    # Use numpy repeat for upsampling (basic array operation)
    up_h = np.repeat(img, 2, axis=0)[:h]
    up_w = np.repeat(up_h, 2, axis=1)[:, :w]
    # Smooth to reduce blockiness
    result = gaussian_blur_manual(up_w, kernel_size=5, sigma=1.0)
    return result


def build_laplacian_pyramid(img, levels):
    """
    Construct a Laplacian pyramid. Each level is the difference
    between consecutive Gaussian pyramid levels (after upsampling
    the coarser level). The final level is the coarsest Gaussian.

    The Laplacian pyramid decomposes the image into frequency bands:
    fine details at the top, coarse structure at the bottom. This
    decomposition is the key to multi-band blending—by blending
    each frequency band independently with different-width masks,
    we avoid visible seams while preserving local contrast.
    """
    gauss_pyr = build_gaussian_pyramid(img, levels)
    lap_pyr = []
    for i in range(levels - 1):
        upsampled = upsample(gauss_pyr[i + 1], gauss_pyr[i].shape)
        lap = gauss_pyr[i] - upsampled
        lap_pyr.append(lap)
    lap_pyr.append(gauss_pyr[-1])  # coarsest level
    return lap_pyr


def multi_band_blend(img1, img2, mask1, mask2, levels=6):
    """
    Multi-band blending using Laplacian pyramids.

    The idea (Burt & Adelson, 1983): decompose both images into
    Laplacian pyramids, build a Gaussian pyramid from the blending
    mask, and combine corresponding levels. Reconstruct from the
    blended Laplacian pyramid. High-frequency details blend with
    a sharp transition (preserving sharpness), while low-frequency
    content blends with a wide, smooth transition (avoiding seams).

    Args:
        img1, img2:   float64 images on the same canvas
        mask1, mask2: float64 masks (0/1) indicating valid pixels
        levels:       number of pyramid levels

    Returns:
        blended: the multi-band blended result
    """
    # Make sure images and masks have the same size
    h, w = img1.shape[:2]

    # Ensure 3-channel for consistency
    if img1.ndim == 2:
        img1 = img1[:, :, None]
    if img2.ndim == 2:
        img2 = img2[:, :, None]

    # Create a smooth weight mask for blending
    # Where both images overlap, create a gradient transition
    overlap = (mask1 > 0.5) & (mask2 > 0.5)
    only1 = (mask1 > 0.5) & (mask2 < 0.5)
    only2 = (mask2 > 0.5) & (mask1 < 0.5)

    # Build the blending weight: use distance transform for smooth transition
    # For the overlap region, weight by relative distance to each image's boundary
    weight1 = np.zeros((h, w), dtype=np.float64)
    weight2 = np.zeros((h, w), dtype=np.float64)

    if np.any(mask1 > 0.5):
        # Distance from non-mask region
        dist1 = cv2.distanceTransform((mask1 > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        weight1 = dist1.astype(np.float64)
    if np.any(mask2 > 0.5):
        dist2 = cv2.distanceTransform((mask2 > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        weight2 = dist2.astype(np.float64)

    # Normalize to get a smooth alpha between 0 and 1
    total = weight1 + weight2
    total[total < 1e-10] = 1e-10
    alpha = weight1 / total  # alpha=1 => use img1, alpha=0 => use img2

    # Expand alpha to 3 channels
    alpha_3 = np.stack([alpha] * 3, axis=-1)

    # Build Laplacian pyramids for both images
    lap1 = build_laplacian_pyramid(img1, levels)
    lap2 = build_laplacian_pyramid(img2, levels)

    # Build Gaussian pyramid for the blending mask
    mask_pyr = build_gaussian_pyramid(alpha_3, levels)

    # Blend each level
    blended_pyr = []
    for l in range(levels):
        blended_level = mask_pyr[l] * lap1[l] + (1.0 - mask_pyr[l]) * lap2[l]
        blended_pyr.append(blended_level)

    # Reconstruct from blended Laplacian pyramid
    result = blended_pyr[-1]
    for l in range(levels - 2, -1, -1):
        upsampled = upsample(result, blended_pyr[l].shape)
        result = upsampled + blended_pyr[l]

    return result


def feathered_blend(img1, img2, mask1, mask2):
    """
    Simpler feathered (distance-weighted) blending as a baseline
    for the "naive vs final" comparison.

    Each pixel in the overlap zone gets a weighted average,
    with weights proportional to the distance from the boundary
    of each image's valid region.
    """
    h, w = img1.shape[:2]
    weight1 = np.zeros((h, w), dtype=np.float64)
    weight2 = np.zeros((h, w), dtype=np.float64)

    if np.any(mask1 > 0.5):
        weight1 = cv2.distanceTransform(
            (mask1 > 0.5).astype(np.uint8), cv2.DIST_L2, 5
        ).astype(np.float64)
    if np.any(mask2 > 0.5):
        weight2 = cv2.distanceTransform(
            (mask2 > 0.5).astype(np.uint8), cv2.DIST_L2, 5
        ).astype(np.float64)

    total = weight1 + weight2
    total[total < 1e-10] = 1e-10

    alpha1 = (weight1 / total)[:, :, None]
    alpha2 = (weight2 / total)[:, :, None]

    blended = alpha1 * img1 + alpha2 * img2
    return blended



# STEP 8: INTENSITY / EXPOSURE COMPENSATION 


def compensate_exposure(img_target, img_source, mask_target, mask_source):
    """
    Simple gain-based exposure compensation.

    In the overlap region between two images, compute the per-channel
    mean intensity ratio and apply it as a multiplicative gain to the
    source image. This corrects for differences in auto-exposure
    between the original photographs.

    Args:
        img_target:   reference image (float64)
        img_source:   image to adjust
        mask_target:  valid-pixel mask for target
        mask_source:  valid-pixel mask for source

    Returns:
        compensated:  exposure-corrected source image
    """
    overlap = (mask_target > 0.5) & (mask_source > 0.5)

    if np.sum(overlap) < 100:
        # Not enough overlap for reliable estimation
        return img_source.copy()

    gains = []
    for c in range(3):
        mean_target = np.mean(img_target[:, :, c][overlap])
        mean_source = np.mean(img_source[:, :, c][overlap])
        if mean_source > 1.0:
            gain = mean_target / mean_source
            # Clamp to avoid extreme corrections
            gain = np.clip(gain, 0.5, 2.0)
        else:
            gain = 1.0
        gains.append(gain)

    compensated = img_source.copy()
    for c in range(3):
        compensated[:, :, c] = compensated[:, :, c] * gains[c]

    print(f"    Exposure gains: R={gains[2]:.3f}, G={gains[1]:.3f}, B={gains[0]:.3f}")
    return compensated


# STEP 9: NAIVE STITCH 


def naive_stitch(warped_images, masks):
    """
    A naive stitch simply overlays images in order, with no blending.
    Later images overwrite earlier ones. This produces visible seams
    at exposure boundaries and ghosting at misalignments.
    """
    canvas = np.zeros_like(warped_images[0])
    for img, mask in zip(warped_images, masks):
        region = mask > 0.5
        if img.ndim == 3:
            for c in range(3):
                canvas[:, :, c][region] = img[:, :, c][region]
        else:
            canvas[region] = img[region]
    return canvas



# STEP 10: FINAL CLEANUP — CROP & RECTANGLE


def crop_to_content(panorama, combined_mask):
    """
    Manual 'Smart Crop' using NumPy Matrix Operations
    
    1. Uses NumPy to find the bounding box of valid pixels .
    2. Uses a 'Greedy Shrink' algorithm to find the largest clean rectangle.
    3. Safety Check: Ensures we don't crop away actual image content.
    """
    # 1. Create a binary mask using NumPy boolean operations
    #    (Strict manual thresholding)
    if combined_mask.dtype == np.float64 or combined_mask.dtype == np.float32:
        # Check for non-black pixels (allow tiny float error > 0.01)
        valid_mask = (combined_mask > 0.01)
    else:
        valid_mask = (combined_mask > 1)

    # 2. Manual Bounding Box (Using NumPy)
    #    Project mask onto axes to find where content starts/ends
    rows = np.any(valid_mask, axis=1)
    cols = np.any(valid_mask, axis=0)
    
    # If image is empty, return original
    if not np.any(rows) or not np.any(cols):
        return panorama

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Cut out the black void
    safe_crop = panorama[y_min:y_max+1, x_min:x_max+1]
    safe_mask = valid_mask[y_min:y_max+1, x_min:x_max+1]

    # Count actual valid pixels (content)
    total_content = np.count_nonzero(safe_mask)

    # 3. Greedy Shrink Loop
    top, bottom = 0, safe_mask.shape[0] - 1
    left, right = 0, safe_mask.shape[1] - 1
    
    # Safety: We will NOT crop if we lose more than 20% of the actual pixels.(for this step, we can allow some cropping, but not if it cuts into the main image content)
    # This prevents the code from returning a tiny square or crashing on slanted images.
    min_content_threshold = total_content * 0.20 # We want to keep at least 20% of the content to avoid over-cropping.(for more wider range of images, we can adjust this threshold as needed)
    
    # Limit iterations to prevent hanging
    max_iters = min(safe_mask.shape[0], safe_mask.shape[1])
    
    for _ in range(max_iters):
        # Current dimensions
        h = bottom - top
        w = right - left
        if h <= 0 or w <= 0: break

        # Get the current ROI from the mask
        roi = safe_mask[top:bottom+1, left:right+1]
        
        # STOP if we are cutting into the main image too much
        if np.count_nonzero(roi) < min_content_threshold:
            print("  [Crop] Stopping to preserve image content.")
            break
            
        # Check the 4 edges using NumPy slicing
        row_top = roi[0, :]
        row_btm = roi[-1, :]
        col_lft = roi[:, 0]
        col_rgt = roi[:, -1]
        
        # Count invalid (False) pixels on edges
        bad_t = np.sum(~row_top) # '~' is logical NOT
        bad_b = np.sum(~row_btm)
        bad_l = np.sum(~col_lft)
        bad_r = np.sum(~col_rgt)

        # If edges are clean, we are done!
        if (bad_t + bad_b + bad_l + bad_r) == 0:
            print(f"  [Crop] Found Clean Rect: {w}x{h} px")
            return safe_crop[top:bottom+1, left:right+1]

        # Greedy choice: Shrink the edge with the most black pixels
        max_bad = max(bad_t, bad_b, bad_l, bad_r)
        
        if max_bad == bad_t:   top += 1
        elif max_bad == bad_b: bottom -= 1
        elif max_bad == bad_l: left += 1
        elif max_bad == bad_r: right -= 1

    # Return the best we found
    return safe_crop[top:bottom+1, left:right+1]

# MAIN 


def memory_safe_linear_blend(img1, img2, mask1, mask2):
    """
    Disk-Based Linear Blending: Uses hard drive (memmap) to store weights.
    This guarantees it will run, even if RAM is completely full.
    """
    print("    -> RAM Full. Switching to Disk-Based Blending (Slow but Robust)...")
    h, w = img1.shape[:2]
    
    # 1. Calculate Weights on Disk
    # We calculate w1, save to disk, delete from RAM. Then w2.
    # This ensures we never hold two huge float arrays in RAM at once.
    
    try:
        # --- Weight 1 ---
        print("       Calculating weight map 1...")
        if np.any(mask1 > 0):
            # mask1 is uint8 255, convert to binary 1
            m1_bin = (mask1 > 127).astype(np.uint8)
            w_ram = cv2.distanceTransform(m1_bin, cv2.DIST_L2, 5)
        else:
            w_ram = np.zeros((h, w), dtype=np.float32)
            
        # Write to disk
        fp1 = np.memmap('temp_w1.dat', dtype='float32', mode='w+', shape=(h, w))
        fp1[:] = w_ram[:]
        del w_ram, m1_bin # Free RAM
        import gc; gc.collect()

        # --- Weight 2 ---
        print("       Calculating weight map 2...")
        if np.any(mask2 > 0):
            m2_bin = (mask2 > 127).astype(np.uint8)
            w_ram = cv2.distanceTransform(m2_bin, cv2.DIST_L2, 5)
        else:
            w_ram = np.zeros((h, w), dtype=np.float32)
            
        # Write to disk
        fp2 = np.memmap('temp_w2.dat', dtype='float32', mode='w+', shape=(h, w))
        fp2[:] = w_ram[:]
        del w_ram, m2_bin
        gc.collect()
        
        # --- Chunked Blending ---
        print("       Blending chunks...")
        # Output array (try RAM, fallback to disk)
        try:
            result = np.zeros(img1.shape, dtype=np.uint8)
        except MemoryError:
            result = np.memmap('temp_result.dat', dtype='uint8', mode='w+', shape=img1.shape)

        CHUNK = 2000
        for y in range(0, h, CHUNK):
            ye = min(y + CHUNK, h)
            
            # Read weights from disk (small slice only)
            w1_c = fp1[y:ye]
            w2_c = fp2[y:ye]
            
            total = w1_c + w2_c
            total[total < 1e-5] = 1e-5
            alpha = (w1_c / total)[:, :, None] # Expand to 3 channels
            
            # Read images
            chunk1 = img1[y:ye].astype(np.float32)
            chunk2 = img2[y:ye].astype(np.float32)
            
            # Blend
            blended = chunk1 * alpha + chunk2 * (1.0 - alpha)
            result[y:ye] = np.clip(blended, 0, 255).astype(np.uint8)

        # Clean up temp files
        del fp1, fp2
        try:
            os.remove('temp_w1.dat')
            os.remove('temp_w2.dat')
        except: pass
        
        return result

    except Exception as e:
        print(f"    [!] Error in disk blending: {e}")
        return img1 # Last resort return

def smart_blend_manager(img1, img2, mask1, mask2, levels=6):
    """
    Smart Blending Manager:
    1. Checks if the image is too big for RAM.
    2. If too big, goes DIRECTLY to Disk-Based Linear Blending (Safe).
    3. If small, tries High-Quality Multi-Band Blending.
    """
    # 1. Estimate Memory Requirement
    # A float64 image takes 8 bytes per pixel per channel.
    # Pyramids create roughly 1.33x copies.
    # We need 2 images + overhead.
    h, w = img1.shape[:2]
    pixels = h * w
    
    # Rough math: (Pixels * 3 channels * 8 bytes) * 2 images * 2 (pyramid overhead)
    # This is a conservative estimate.
    est_mem_bytes = (pixels * 3 * 8) * 4 
    est_mem_gb = est_mem_bytes / (1024**3)
    
    print(f"  [Memory Check] Image size: {w}x{h}")
    print(f"  [Memory Check] Est. Multi-Band RAM needed: ~{est_mem_gb:.2f} GB")
    
    # THRESHOLD: If it needs more than 2.0 GB, skip Multi-Band entirely.
    # This prevents the C++ OpenCV crash.
    SAFE_LIMIT_GB = 2.0 
    
    if est_mem_gb > SAFE_LIMIT_GB: 
        print(f"  [!] Image exceeds safe limit ({SAFE_LIMIT_GB} GB). Forcing Disk-Based Blending.")
        return memory_safe_linear_blend(img1, img2, mask1, mask2)

    # 2. Try High-Quality Blend (only for small images)
    try:
        return multi_band_blend(img1.astype(np.float64), img2.astype(np.float64), 
                                mask1, mask2, levels)
    except (MemoryError, np.core._exceptions._ArrayMemoryError, cv2.error):
        # Catch both Python MemoryErrors and OpenCV C++ Errors
        print("\n  [!] RAM Full or OpenCV Error! Switching to Disk-Based Blending...")
        import gc
        gc.collect()
        return memory_safe_linear_blend(img1, img2, mask1, mask2)



def stitch_panorama(image_paths, output_dir="output"):
    """
    Full panoramic stitching pipeline.

    Strategy: Use the CENTER image as the reference frame (identity
    homography). Compute homographies that map the left image -> center
    and right image -> center. For more than 3 images, chain
    homographies outward from center.

    This center-projection approach minimizes the total amount of
    perspective distortion in the final panorama, since the center
    image undergoes no warping at all.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PANORAMIC IMAGE STITCHING ")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load images
    # ------------------------------------------------------------------
    print("\n[1/8] Loading images...")
    images_bgr, images_gray = load_images(image_paths)
    n_images = len(images_bgr)
    display_images(images_bgr)

    # The center image is our reference
    center_idx = n_images // 2
    print(f"  Reference (center) image: index {center_idx}")

    # ------------------------------------------------------------------
    # 2. Extract SIFT features
    # ------------------------------------------------------------------
    print("\n[2/8] Extracting SIFT features...")
    keypoints, descriptors = extract_sift_features(images_gray)
    visualize_keypoints(images_bgr, keypoints)

    # ------------------------------------------------------------------
    # 3-5. For each adjacent pair, match features & compute homography
    # ------------------------------------------------------------------
    print("\n[3/8] Matching features & computing homographies...")

    # We'll store homographies mapping each image -> center reference frame
    # H[center_idx] = Identity
    homographies = [None] * n_images
    homographies[center_idx] = np.eye(3)

    # Process pairs outward from center
    # Left side: center-1, center-2, ...
    for i in range(center_idx - 1, -1, -1):
        print(f"\n  Pair: Image {i} <-> Image {i+1}")

        # Match features
        matches = match_features(descriptors[i], descriptors[i + 1])
        pts_i, pts_ip1 = get_matched_points(keypoints[i], keypoints[i + 1], matches)

        # Visualize matches
        visualize_matches(
            images_bgr[i], keypoints[i],
            images_bgr[i + 1], keypoints[i + 1],
            matches,
            title=f"Matches: Image {i} <-> Image {i+1}",
            fname=f"03_matches_{i}_{i+1}.png"
        )

        # Compute homography: Image i -> Image i+1
        H_i_to_ip1, inlier_mask = compute_homography_ransac(pts_i, pts_ip1)

        # Chain to get Image i -> center
        # H[i] = H[i+1] @ H_i_to_ip1
        homographies[i] = homographies[i + 1] @ H_i_to_ip1
        print(f"    Homography (Image {i} -> center) computed successfully")

    # Right side: center+1, center+2, ...
    for i in range(center_idx + 1, n_images):
        print(f"\n  Pair: Image {i} <-> Image {i-1}")

        # Match features
        matches = match_features(descriptors[i], descriptors[i - 1])
        pts_i, pts_im1 = get_matched_points(keypoints[i], keypoints[i - 1], matches)

        # Visualize matches
        visualize_matches(
            images_bgr[i], keypoints[i],
            images_bgr[i - 1], keypoints[i - 1],
            matches,
            title=f"Matches: Image {i} <-> Image {i-1}",
            fname=f"03_matches_{i}_{i-1}.png"
        )

        # Compute homography: Image i -> Image i-1
        H_i_to_im1, inlier_mask = compute_homography_ransac(pts_i, pts_im1)

        # Chain: H[i] = H[i-1] @ H_i_to_im1
        homographies[i] = homographies[i - 1] @ H_i_to_im1
        print(f"    Homography (Image {i} -> center) computed successfully")

    # ------------------------------------------------------------------
    # 6. Compute canvas size and warp all images
    # ------------------------------------------------------------------
    print("\n[4/8] Computing output canvas bounds...")

    all_x_min, all_y_min = 0, 0
    all_x_max, all_y_max = 0, 0

    for i in range(n_images):
        h, w = images_bgr[i].shape[:2]
        xmin, ymin, xmax, ymax = compute_warped_bounds(homographies[i], h, w)
        all_x_min = min(all_x_min, xmin)
        all_y_min = min(all_y_min, ymin)
        all_x_max = max(all_x_max, xmax)
        all_y_max = max(all_y_max, ymax)

    out_w = all_x_max - all_x_min + 1
    out_h = all_y_max - all_y_min + 1
    offset = (all_x_min, all_y_min)  # translation offset
    print(f"  Canvas size: {out_w} x {out_h} px, offset: {offset}")

    print("\n[5/8] Warping images into common coordinate frame...")
    warped_images = []
    masks = []
    for i in range(n_images):
        print(f"  Warping image {i}...")
        warped, mask = warp_image(
            images_bgr[i], homographies[i],
            output_size=(out_h, out_w),
            offset=offset
        )
        
        # --- CRITICAL FIX: Save 85% RAM immediately ---
        # Convert float64 (8 bytes/pixel) -> uint8 (1 byte/pixel)
        # This reduces a 8GB image to 1GB.
        warped_u8 = np.clip(warped, 0, 255).astype(np.uint8)
        warped_images.append(warped_u8)
        
        # Save mask as uint8 (0 or 255) to save space too
        mask_u8 = (mask > 0.5).astype(np.uint8) * 255
        masks.append(mask_u8)
        
        # Force delete the heavy arrays
        del warped, mask
        import gc; gc.collect()

    # ------------------------------------------------------------------
    # 7. Exposure compensation
    # ------------------------------------------------------------------


    print("\n[6/8] Compensating exposure differences...")
    for i in range(n_images):
        if i == center_idx:
            continue
        
        # Convert to float temporarily for math
        target_f = warped_images[center_idx].astype(np.float64)
        source_f = warped_images[i].astype(np.float64)
        
        # Masks are now uint8 0-255, normalize to 0-1 for the function
        mask_t = masks[center_idx] / 255.0
        mask_s = masks[i] / 255.0
        
        compensated = compensate_exposure(
            target_f, source_f,
            mask_t, mask_s
        )
        
        # Convert back to uint8 immediately to save RAM
        warped_images[i] = np.clip(compensated, 0, 255).astype(np.uint8)
        
        # Clean up
        del target_f, source_f, compensated
        gc.collect()

    # ------------------------------------------------------------------
    # 8a. Naive stitch (for report comparison)
    # ------------------------------------------------------------------
    print("\n[7/8] Creating naive stitch (direct overlay, no blending)...")
    naive = naive_stitch(warped_images, masks)
    naive_clipped = np.clip(naive, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "04_naive_stitch.png"), naive_clipped)

    # ------------------------------------------------------------------
    # 8b. Multi-band blending (final panorama)
    # ------------------------------------------------------------------
    print("\n[8/8] Multi-band Laplacian blending for seamless result...")

    # Iteratively blend: start with center, blend outward
    blended = warped_images[center_idx].copy()
    blended_mask = masks[center_idx].copy()

    # Order of blending: alternate left and right from center
    blend_order = []
    for d in range(1, n_images):
        if center_idx - d >= 0:
            blend_order.append(center_idx - d)
        if center_idx + d < n_images:
            blend_order.append(center_idx + d)

    for idx in blend_order:
        print(f"  Blending image {idx} into panorama...")
        
        # Ensure we are passing valid types
        if blended.dtype != np.uint8:
            blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        curr_img = np.clip(warped_images[idx], 0, 255).astype(np.uint8)

        # Use the MANAGER instead of calling multi_band_blend directly
        blended = smart_blend_manager(
            blended, curr_img,
            blended_mask, masks[idx],
            levels=6
        )
        # Update combined mask
        blended_mask = np.maximum(blended_mask, masks[idx])

    blended_clipped = np.clip(blended, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # 9. Final cleanup — crop to clean rectangle
    # ------------------------------------------------------------------
    print("\n  Cropping to clean rectangle...")
    final_panorama = crop_to_content(blended_clipped, blended_mask)
    naive_cropped = crop_to_content(naive_clipped, blended_mask)

    # Save results
    cv2.imwrite(os.path.join(output_dir, "05_blended_panorama.png"), final_panorama)
    cv2.imwrite(os.path.join(output_dir, "04_naive_stitch_cropped.png"), naive_cropped)

    # ------------------------------------------------------------------
    # 10. Comparison figure for the report
    # ------------------------------------------------------------------
    print("\n  Generating comparison figure...")
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    axes[0].imshow(cv2.cvtColor(naive_cropped, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Naive Stitch (Direct Overlay — No Blending)", fontsize=13)
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Final Panorama (Multi-Band Laplacian Blending + Exposure Compensation)", fontsize=13)
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 70)
    print("Done All are saved to:", output_dir)
    print("=" * 70)
    print("  01_input_images.png          — original images side by side")
    print("  02_sift_keypoints.png        — detected SIFT keypoints")
    print("  03_matches_*.png             — feature matches per pair")
    print("  04_naive_stitch.png          — naive direct-overlay stitch")
    print("  05_blended_panorama.png      — final multi-band blended panorama")
    print("  06_comparison.png            — side-by-side naive vs final")

    return final_panorama




if __name__ == "__main__":
    """
    USAGE:
        python panorama_stitcher.py img_left.jpg img_center.jpg img_right.jpg

    The images should be supplied left-to-right in the order they
    were captured. The middle image will be used as the reference.
    You may supply 3 or more images.
    """

    if len(sys.argv) < 4:
        print("Usage: python panorama_stitcher.py <left.jpg> <center.jpg> <right.jpg> [more...]")
        print("Supply at least 3 overlapping images, ordered left to right.")
        sys.exit(1)

    image_paths = sys.argv[1:]
    print(f"Stitching {len(image_paths)} images...")
    stitch_panorama(image_paths, output_dir="output")