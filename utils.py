import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.spatial.transform import Rotation as R

##############################################################################################################
def find_corners(left_dir, right_dir, left_img_list, right_img_list, objp, chessboard_size):
    """
    Detect chessboard corners in synchronized left/right images for stereo calibration.

    Args:
        left_dir (str): Directory containing left camera images.
        right_dir (str): Directory containing right camera images.
        left_img_list (list[str]): Filenames of left images (order must match right images).
        right_img_list (list[str]): Filenames of right images (order must match left images).
        objp (np.ndarray): (N, 3) array of chessboard corner coordinates in the world frame.
        chessboard_size (tuple[int, int]): Number of inner corners per chessboard (cols, rows).

    Returns:
        imgpoints_L (list[np.ndarray]): Per-image list of detected 2D corners in left images (N_i,1,2).
        imgpoints_R (list[np.ndarray]): Per-image list of detected 2D corners in right images (N_i,1,2).
        objpoints (list[np.ndarray]): Per-image list of corresponding 3D points (N_i,3) in world frame.
        success_frames (list[list[str]]): List of [left_filename, right_filename] pairs that succeeded.
    """
    objpoints = []                    # 3D points (one array per successful stereo pair)
    imgpoints_L = []                  # 2D detections in the left images
    imgpoints_R = []                  # 2D detections in the right images

    num_success = 0
    success_frames = []
    num_fail = 0

    for img_name_L, img_name_R in zip(left_img_list, right_img_list):
        # Build full paths and load images
        path_L = os.path.join(left_dir, img_name_L)
        path_R = os.path.join(right_dir, img_name_R)
        img_L = cv2.imread(path_L)
        img_R = cv2.imread(path_R)

        # Convert to grayscale for detection
        gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

        # Robust checkerboard detection (SB variant)
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ret_L, corners_L = cv2.findChessboardCornersSB(gray_L, chessboard_size, flags)
        ret_R, corners_R = cv2.findChessboardCornersSB(gray_R, chessboard_size, flags)

        if ret_L and ret_R:
            # Only accept when both sides succeed
            imgpoints_L.append(corners_L)
            imgpoints_R.append(corners_R)
            objpoints.append(objp)
            num_success += 1
            success_frames.append([img_name_L, img_name_R])
        else:
            num_fail += 1

    cv2.destroyAllWindows()
    return imgpoints_L, imgpoints_R, objpoints, success_frames

def stereo_calibration(objpoints, imgpoints_L, imgpoints_R, img_size,
                       K_L=None, D_L=None, K_R=None, D_R=None,
                       criteria=None, flags=0):
    """
    Run joint stereo calibration and compute rectification transforms.

    Args:
        objpoints (list[np.ndarray]): 3D chessboard points in world coordinates per view.
        imgpoints_L (list[np.ndarray]): 2D detected corners in left images per view.
        imgpoints_R (list[np.ndarray]): 2D detected corners in right images per view.
        img_size (tuple[int, int]): Image size as (width, height).
        K_L (np.ndarray|None): Initial/known left intrinsics (3x3) or None to estimate.
        D_L (np.ndarray|None): Initial/known left distortion coeffs or None to estimate.
        K_R (np.ndarray|None): Initial/known right intrinsics (3x3) or None to estimate.
        D_R (np.ndarray|None): Initial/known right distortion coeffs or None to estimate.
        criteria (tuple|None): Termination criteria for the optimizer.
        flags (int): Stereo calibration flags (e.g., cv2.CALIB_FIX_ASPECT_RATIO).

    Returns:
        dict: Stereo parameters with the following keys:
            'K_L','D_L','K_R','D_R' : intrinsics and distortion for left/right
            'R','T'                 : rotation and translation from left to right
            'E','F'                 : essential and fundamental matrices
            'R_L','R_R'             : rectification rotations for left/right
            'P_L','P_R'             : new projection matrices for left/right
            'Q'                     : disparity-to-depth re-projection matrix
    """
    # Joint stereo calibration (refines intrinsics, extrinsics)
    ret, K_L, D_L, K_R, D_R, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_L, imgpoints_R,
        K_L, D_L, K_R, D_R,
        img_size,
        criteria=criteria,
        flags=flags
    )

    # Compute rectification transforms (align epipolar lines)
    R_L, R_R, P_L, P_R, Q, ROI1, ROI2 = cv2.stereoRectify(
        K_L, D_L, K_R, D_R, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
    )

    # Bundle results
    stereo_params = {
        'K_L': K_L, 'D_L': D_L,
        'K_R': K_R, 'D_R': D_R,
        'R': R, 'T': T,
        'E': E, 'F': F,
        'R_L': R_L, 'R_R': R_R,
        'P_L': P_L, 'P_R': P_R,
        'Q': Q
    }
    return stereo_params

###############################################################################################################
def rmse_2D_detection(objpoints, imgpoints, K, D):
    """
    Compute 2D reprojection RMSE per view (in pixels and mm) using solvePnP + projectPoints.

    For each view, estimates the pose (Rvec, Tvec) from detected corners,
    projects the known 3D points back to the image, and measures RMSE in pixels.
    Also converts pixel error to millimeters at each point's depth using fx, fy.

    Args:
        objpoints (list[np.ndarray]): 3D chessboard points per view (N_i, 3).
        imgpoints (list[np.ndarray]): 2D detected corners per view (N_i, 1, 2).
        K (np.ndarray): Camera intrinsic matrix (3x3).
        D (np.ndarray): Distortion coefficients.

    Returns:
        all_rmse_px (list[float]): Per-view 2D RMSE in pixels.
        all_rmse_mm (list[float]): Per-view 2D RMSE converted to millimeters.
        dist (list[float]): Per-view Z distance (Tvec[2]) in millimeters.
    """
    all_rmse_px = []
    all_rmse_mm = []
    dist = []

    for i in range(len(objpoints)):
        # Pose from 2D-3D correspondences
        _, Rvec, Tvec = cv2.solvePnP(objpoints[i], imgpoints[i], K, D)

        # Reproject 3D points back to 2D
        proj, _ = cv2.projectPoints(objpoints[i], Rvec, Tvec, K, D)

        # ---------- Pixel RMSE ----------
        real_px = np.asarray(imgpoints[i], dtype=np.float32)
        pred_px = np.asarray(proj, dtype=np.float32)
        err_px = real_px - pred_px               # (N,1,2)
        err_px = err_px.reshape(-1, 2)           # (N,2)
        rmse_px = np.sqrt(np.mean(err_px[:, 0]**2 + err_px[:, 1]**2))
        all_rmse_px.append(rmse_px)

        # ---------- Millimeter RMSE ----------
        fx, fy = K[0, 0], K[1, 1]

        # Compute per-point depths in the camera frame
        Rot, _ = cv2.Rodrigues(Rvec)
        Tran = Tvec
        obj_pts = objpoints[i].reshape(-1, 3)            # (N,3)
        pts_cam = (Rot @ obj_pts.T + Tran).T             # (N,3)
        Z_vals = pts_cam[:, 2]                           # depth in mm

        # Convert pixel error to metric error at each depth
        err_mm_x = (err_px[:, 0] * Z_vals / fx)
        err_mm_y = (err_px[:, 1] * Z_vals / fy)
        rmse_mm = np.sqrt(np.mean(err_mm_x**2 + err_mm_y**2))
        all_rmse_mm.append(rmse_mm)

        # Store working distance (camera-to-target Z)
        dist.append(float(Tvec[2]))

    return all_rmse_px, all_rmse_mm, dist

#####################################################################################################
def plot_rmse_dist(dist, rmse, frame_names, unit, side):
    """
    Scatter-plot RMSE versus working distance with Pearson correlation annotation.

    Args:
        dist (list[float]): Per-view working distance (e.g., Tvec[2]) in mm.
        rmse (list[float]): Per-view RMSE values to plot (same length as dist).
        frame_names (list[str]): Labels for each point (e.g., image/frame names).
        unit (str): Unit label to append to the RMSE axis title (e.g., 'px' or 'mm').
        side (str): Short camera label for figure title (e.g., 'Left ' or 'Right ').

    Returns:
        None (renders a Matplotlib figure).
    """
    from scipy.stats import pearsonr
    import matplotlib.pyplot as plt

    # Pearson correlation between distance and error
    corr, pval = pearsonr(dist, rmse)

    plt.figure(figsize=(10, 4))
    plt.scatter(dist, rmse)

    # Label each point with its frame name
    for x, y, name in zip(dist, rmse, frame_names):
        plt.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    plt.xlabel('Working Distance (mm)')
    plt.ylabel('Reprojection RMSE ' + unit)
    plt.title(side + f'Camera\nCorrelation: {corr:.3f}' + f'  p-value: {pval:.3f}')
    plt.grid(True)


##############################################################################################
def undistort_rectify_save(img, save_path, map1, map2):
    """
    Apply precomputed rectification/undistortion maps and save the rectified image.

    Args:
        img (np.ndarray): Input image (distorted).
        save_path (str): Output path to save the remapped (rectified) image.
        map1 (np.ndarray): x-map from cv2.initUndistortRectifyMap / cv2.fisheye.initUndistortRectifyMap.
        map2 (np.ndarray): y-map from cv2.initUndistortRectifyMap / cv2.fisheye.initUndistortRectifyMap.

    Returns:
        None (writes image to disk).
    """
    # Remap using bilinear interpolation
    img_rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(save_path, img_rect)

##############################################################################################
def quaternion_to_transformation(quaternion):
    """
    Convert a quaternion + translation to a 4x4 homogeneous transformation matrix.

    Note:
        This function expects a mapping-like input (e.g., dict, pandas Series, or 1-row DataFrame slice)
        containing keys: 'q0','qx','qy','qz','tx','ty','tz'. The quaternion is treated with scalar first.

    Args:
        quaternion: A mapping with fields q0, qx, qy, qz, tx, ty, tz (floats).

    Returns:
        np.ndarray: 4x4 homogeneous transform with rotation from the quaternion and translation (tx,ty,tz).
    """
    from scipy.spatial.transform import Rotation as R

    # Extract quaternion and translation; ensure float
    q0, qx, qy, qz = float(quaternion['q0']), float(quaternion['qx']), float(quaternion['qy']), float(quaternion['qz'])
    tx, ty, tz = float(quaternion['tx']), float(quaternion['ty']), float(quaternion['tz'])

    # Quaternion to rotation (scalar-first convention)
    rotation = R.from_quat([q0, qx, qy, qz], scalar_first=True)
    rotation_matrix = rotation.as_matrix()  # 3x3 rotation

    # Build 4x4 homogeneous transform
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]
    return transformation_matrix

######################################################################################################
def calculate_BF_to_EN(polaris_readings, objpoints, imgpoints, frame_names, CPcb_to_CPot, K, D):
    """
    Compute BF_to_EN (endoscope body frame to endoscope optical frame) per frame,
    using optical tracking data and PnP-estimated EN_to_CPcb.

    For each frame:
      - Read OT poses for Endoscope (E) and Calibration Plate (C).
      - Convert them to transforms, chain them to get BF_to_CPcb.
      - Estimate EN_to_CPcb from image corners (solvePnP).
      - Invert to get CPcb_to_EN, then BF_to_EN = BF_to_CPcb * CPcb_to_EN.

    Args:
        polaris_readings (pd.DataFrame): Tracking rows with columns including:
            ' frame name', 'Tool Type' (E/C), and pose fields q0,qx,qy,qz,tx,ty,tz.
        objpoints (list[np.ndarray]): 3D chessboard points per frame (N_i,3).
        imgpoints (list[np.ndarray]): 2D chessboard detections per frame (N_i,1,2).
        frame_names (list[str]): Ordered list of frame filenames to process.
        CPcb_to_CPot (np.ndarray): 4x4 transform from calibration plate CB coordinate to OT plate coordinate.
        K (np.ndarray): Camera intrinsics (3x3).
        D (np.ndarray): Distortion coefficients.

    Returns:
        BF_to_EN_list (list[np.ndarray]): Per-frame 4x4 BF_to_EN transforms.
        CPcb_to_EN_list (list[np.ndarray]): Per-frame 4x4 CPcb_to_EN transforms.
        EN_to_CPcb_list (list[np.ndarray]): Per-frame 4x4 EN_to_CPcb transforms (from PnP).
        BF_to_CPcb_list (list[np.ndarray]): Per-frame 4x4 BF_to_CPcb transforms (from OT chain).
    """
    BF_to_EN_list = []
    EN_to_CPcb_list = []
    CPcb_to_EN_list = []
    BF_to_CPcb_list = []

    idx = 0
    for frame in frame_names:
        # Filter tracking rows for this frame
        temp = polaris_readings[polaris_readings[' frame name'] == frame]

        # E: endoscope base (OT_to_BF), C: calibration plate (OT_to_CPot)
        OT_to_BF_raw = temp[temp['Tool Type'] == 'E']
        OT_to_CPot_raw = temp[temp['Tool Type'] == 'C']

        # Extract quaternion+translation fields
        OT_to_BF_qt = OT_to_BF_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]
        OT_to_CPot_qt = OT_to_CPot_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]

        # Convert to 4x4 transforms
        OT_to_BF = quaternion_to_transformation(OT_to_BF_qt)
        OT_to_CPot = quaternion_to_transformation(OT_to_CPot_qt)

        # Chain: BF<-OT and CPcb<-CPot
        BF_to_OT = np.linalg.inv(OT_to_BF)
        BF_to_CPot = np.dot(BF_to_OT, OT_to_CPot)
        CPot_to_CPcb = np.linalg.inv(CPcb_to_CPot)
        BF_to_CPcb = np.dot(BF_to_CPot, CPot_to_CPcb)
        BF_to_CPcb_list.append(BF_to_CPcb)

        # Estimate EN_to_CPcb from image detections (solvePnP)
        _, Rvec, Tvec = cv2.solvePnP(objpoints[idx], imgpoints[idx], K, D)
        temp_Rot, _ = cv2.Rodrigues(Rvec)

        EN_to_CPcb = np.eye(4)
        EN_to_CPcb[:3, :3] = temp_Rot
        EN_to_CPcb[:3, 3] = Tvec.flatten()
        EN_to_CPcb_list.append(EN_to_CPcb)

        # Invert to get CPcb_to_EN, then compute BF_to_EN
        CPcb_to_EN = np.linalg.inv(EN_to_CPcb)
        CPcb_to_EN_list.append(CPcb_to_EN)
        BF_to_EN = np.dot(BF_to_CPcb, CPcb_to_EN)
        BF_to_EN_list.append(BF_to_EN)

        idx += 1

    return BF_to_EN_list, CPcb_to_EN_list, EN_to_CPcb_list, BF_to_CPcb_list


#######################################################################################################
def average_rotation_matrix_svd(R_list):
    """
    Compute an average rotation from a list of rotation matrices using SVD (projected averaging).

    Args:
        R_list (list[np.ndarray]): List of 3x3 rotation matrices.

    Returns:
        np.ndarray: Averaged 3x3 rotation matrix (guaranteed to be a valid rotation).
    """
    # Sum all rotation matrices
    M = np.zeros((3, 3))
    for R_i in R_list:
        M += R_i

    # SVD and projection back to SO(3)
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt

    # Ensure a proper rotation (determinant > 0)
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg

def estimate_transform_average(B_list, C_list):
    """
    Estimate the average rigid transform A that best matches A ≈ B_i * C_i over i.

    Strategy:
      - Form A_i = B_i * C_i for each i.
      - Average translations directly.
      - Average rotations via SVD-based projection.

    Args:
        B_list (list[np.ndarray]): List of 4x4 transforms.
        C_list (list[np.ndarray]): List of 4x4 transforms.

    Returns:
        np.ndarray: 4x4 averaged transform A_avg.

    Raises:
        AssertionError: If list lengths differ.
    """
    assert len(B_list) == len(C_list), "B_list and C_list must have the same length."
    N = len(B_list)
    A_list = [B_list[i] @ C_list[i] for i in range(N)]

    # Translation: arithmetic mean
    t_list = np.array([A[:3, 3] for A in A_list])
    t_avg = np.mean(t_list, axis=0)

    # Rotation: SVD-based averaging
    R_list = [A[:3, :3] for A in A_list]
    R_avg = average_rotation_matrix_svd(R_list)

    # Compose final 4x4 rigid transform
    A_avg = np.eye(4)
    A_avg[:3, :3] = R_avg
    A_avg[:3, 3] = t_avg
    return A_avg


#######################################################################################################
def rmse_2D_transform(polaris_readings, objp, objpoints, imgpoints, chessboard_size, img_dir, frame_names, CPcb_to_CPot, BF_to_EN, K, D, save_img = False, save_dir=None):
    """
    Compute 2D reprojection RMSE for a known chain BF->EN and tracked OT poses.

    For each frame:
      - Build EN_to_CPcb from the chain (BF_to_EN, OT poses, CPot_to_CPcb).
      - Project the chessboard model to the image and compute 2D RMSE (px and mm).
      - Optionally draw and save projected corners over the image.

    Args:
        polaris_readings (pd.DataFrame): Tracking table with columns as in calculate_BF_to_EN.
        objp (np.ndarray): (N,3) chessboard points (single board model).
        objpoints (list[np.ndarray]): Per-frame 3D points (for mm conversion via solvePnP).
        imgpoints (list[np.ndarray]): Per-frame detected 2D corners.
        chessboard_size (tuple[int,int]): Chessboard inner corners (cols, rows) for drawing.
        img_dir (str): Directory containing raw images.
        frame_names (list[str]): Filenames to process, matching polaris_readings and imgpoints.
        CPcb_to_CPot (np.ndarray): 4x4 transform CPcb<-CPot.
        BF_to_EN (np.ndarray): 4x4 transform EN<-BF (fixed/estimated).
        K (np.ndarray): Intrinsic matrix.
        D (np.ndarray): Distortion coefficients.
        save_img (bool): If True, save visualization images with projected corners.
        save_dir (str|None): Where to save the visualizations (required if save_img=True).

    Returns:
        all_rmse_px (list[float]): Per-frame reprojection RMSE in pixels.
        all_rmse_mm (list[float]): Per-frame reprojection RMSE in millimeters.
        dist (list[float]): Per-frame Z distance (from solvePnP) in millimeters.
    """
    idx = 0
    all_rmse_px = []
    all_rmse_mm = []
    dist = []

    for frame in frame_names:
        # Read tracking rows for this frame
        temp = polaris_readings[polaris_readings[' frame name'] == frame]
        img = cv2.imread(os.path.join(img_dir, frame))

        # E: endoscope base; C: calibration plate
        OT_to_BF_raw = temp[temp['Tool Type'] == 'E']
        OT_to_CPot_raw = temp[temp['Tool Type'] == 'C']

        # Extract pose fields
        OT_to_BF_qt = OT_to_BF_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]
        OT_to_CPot_qt = OT_to_CPot_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]

        # Convert to transforms
        OT_to_BF = quaternion_to_transformation(OT_to_BF_qt)
        OT_to_CPot = quaternion_to_transformation(OT_to_CPot_qt)

        # Chain to get EN_to_CPcb
        BF_to_OT = np.linalg.inv(OT_to_BF)
        BF_to_CPot = np.dot(BF_to_OT, OT_to_CPot)
        EN_to_BF = np.linalg.inv(BF_to_EN)
        EN_to_CPot = np.dot(EN_to_BF, BF_to_CPot)
        CPot_to_CPcb = np.linalg.inv(CPcb_to_CPot)
        EN_to_CPcb = np.dot(EN_to_CPot, CPot_to_CPcb)

        # Separate rotation & translation for projection
        Rot_EN_to_CPcb = EN_to_CPcb[:3, :3]
        Tran_EN_to_CPcb = EN_to_CPcb[:3, 3]

        # cv2.projectPoints expects Rodrigues rotation vector
        Rot, _ = cv2.Rodrigues(Rot_EN_to_CPcb)
        Tran = Tran_EN_to_CPcb
        projected_points, _ = cv2.projectPoints(objp, Rot, Tran, K, D)

        # Optionally draw and save overlay
        if save_img:
            cv2.drawChessboardCorners(img, chessboard_size, projected_points, True)
            save_path = os.path.join(save_dir, frame)
            cv2.imwrite(save_path, img)

        # ---------- Pixel RMSE ----------
        real_pts = np.asarray(imgpoints[idx], dtype=np.float32)
        pred_pts = np.asarray(projected_points, dtype=np.float32)
        err_px = real_pts - pred_pts
        err_px = err_px.reshape(-1, 2)
        rmse_px = np.sqrt(np.mean(np.sum(err_px**2, axis=1)))
        all_rmse_px.append(rmse_px)

        # ---------- Millimeter RMSE ----------
        fx = K[0, 0]
        fy = K[1, 1]

        # Use solvePnP on detections to estimate z-depths for px->mm conversion
        _, Rvec, Tvec = cv2.solvePnP(objpoints[idx], imgpoints[idx], K, D)
        Rot, _ = cv2.Rodrigues(Rvec)
        Tran = Tvec
        obj_pts = objpoints[idx].reshape(-1, 3)
        pts_cam = (Rot @ obj_pts.T + Tran).T
        Z_vals = pts_cam[:, 2]  # depth in mm

        err_mm_x = (err_px[:, 0] * Z_vals / fx)
        err_mm_y = (err_px[:, 1] * Z_vals / fy)
        rmse_mm = np.sqrt(np.mean(err_mm_x**2 + err_mm_y**2))
        all_rmse_mm.append(rmse_mm)

        # Store working distance
        dist.append(float(Tvec[2]))
        idx += 1

    return all_rmse_px, all_rmse_mm, dist

#######################################################################################################
def rmse_3D_transform(polaris_readings, objpoints, imgpoints, img_dir, frame_names, CPcb_to_CPot, BF_to_EN, K, D):
    """
    Compute 3D RMSE between predicted 3D chessboard points (via EN_to_CPcb chain)
    and ground truth 3D points estimated by solvePnP for each frame.

    Args:
        polaris_readings (pd.DataFrame): Tracking table with OT poses.
        objpoints (list[np.ndarray]): Per-frame 3D chessboard points (N_i,3).
        imgpoints (list[np.ndarray]): Per-frame 2D detections (N_i,1,2).
        img_dir (str): Directory containing raw images (not strictly used in this metric).
        frame_names (list[str]): Filenames/IDs to iterate in order.
        CPcb_to_CPot (np.ndarray): 4x4 transform CPcb<-CPot.
        BF_to_EN (np.ndarray): 4x4 transform EN<-BF.
        K (np.ndarray): Camera intrinsics.
        D (np.ndarray): Distortion coefficients.

    Returns:
        all_rmse_3D (list[float]): Per-frame 3D RMSE in mm.
        dist (list[float]): Per-frame Z distance from PnP (mm).
    """
    idx = 0
    all_rmse_3D = []
    dist = []

    for frame in frame_names:
        # Filter OT rows for this frame
        temp = polaris_readings[polaris_readings[' frame name'] == frame]
        img = cv2.imread(os.path.join(img_dir, frame))  # loaded but not required for metric

        # E: endoscope base; C: calibration plate
        OT_to_BF_raw = temp[temp['Tool Type'] == 'E']
        OT_to_CPot_raw = temp[temp['Tool Type'] == 'C']

        # Extract and convert to transforms
        OT_to_BF_qt = OT_to_BF_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]
        OT_to_CPot_qt = OT_to_CPot_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]
        OT_to_BF = quaternion_to_transformation(OT_to_BF_qt)
        OT_to_CPot = quaternion_to_transformation(OT_to_CPot_qt)

        # Chain to get EN_to_CPcb
        BF_to_OT = np.linalg.inv(OT_to_BF)
        BF_to_CPot = np.dot(BF_to_OT, OT_to_CPot)
        EN_to_BF = np.linalg.inv(BF_to_EN)
        EN_to_CPot = np.dot(EN_to_BF, BF_to_CPot)
        CPot_to_CPcb = np.linalg.inv(CPcb_to_CPot)
        EN_to_CPcb = np.dot(EN_to_CPot, CPot_to_CPcb)

        # Predict 3D points in EN frame
        Rot_EN_to_CPcb = EN_to_CPcb[:3, :3]
        Tran_EN_to_CPcb = EN_to_CPcb[:3, 3]
        obj_pts = objpoints[idx].reshape(-1, 3).astype(np.float64)
        pts_pred = (Rot_EN_to_CPcb @ obj_pts.T + Tran_EN_to_CPcb.reshape(3, 1)).T  # (N,3)

        # Ground truth 3D via solvePnP pose (projective consistency)
        _, Rvec, Tvec = cv2.solvePnP(objpoints[idx], imgpoints[idx], K, D)
        Rvec, _ = cv2.Rodrigues(Rvec)
        Tvec = Tvec.reshape(3)
        pts_gt = (Rvec @ obj_pts.T + Tvec.reshape(3, 1)).T  # (N,3)

        # 3D RMSE (in mm)
        err = pts_pred - pts_gt
        rmse_3d = np.sqrt(np.mean(err[:, 0]**2 + err[:, 1]**2 + err[:, 2]**2))
        all_rmse_3D.append(float(rmse_3d))

        # Optionally track working distance (e.g., Z of GT pose)
        dist.append(float(Tvec[2]))
        idx += 1

    return all_rmse_3D, dist

#######################################################################################################
def rmse_3D_transform_plot(polaris_readings, objpoints, imgpoints, img_dir, frame_names, CPcb_to_CPot, BF_to_EN, K, D, plot=True):
    """
    Same as rmse_3D_transform, with optional 3D scatter visualization per frame.

    Args:
        polaris_readings, objpoints, imgpoints, img_dir, frame_names, CPcb_to_CPot, BF_to_EN, K, D: See rmse_3D_transform.
        plot (bool): If True, show 3D GT vs Pred scatter and segment errors.

    Returns:
        all_rmse_3d (list[float]): Per-frame 3D RMSE in mm.
    """
    idx = 0
    all_rmse_3d = []

    for frame in frame_names:
        # Per-frame OT rows
        temp = polaris_readings[polaris_readings[' frame name'] == frame]

        OT_to_BF_raw = temp[temp['Tool Type'] == 'E']
        OT_to_CPot_raw = temp[temp['Tool Type'] == 'C']

        OT_to_BF_qt = OT_to_BF_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]
        OT_to_CPot_qt = OT_to_CPot_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]

        OT_to_BF = quaternion_to_transformation(OT_to_BF_qt)
        OT_to_CPot = quaternion_to_transformation(OT_to_CPot_qt)

        BF_to_OT = np.linalg.inv(OT_to_BF)
        BF_to_CPot = np.dot(BF_to_OT, OT_to_CPot)

        EN_to_BF = np.linalg.inv(BF_to_EN)
        EN_to_CPot = np.dot(EN_to_BF, BF_to_CPot)
        CPot_to_CPcb = np.linalg.inv(CPcb_to_CPot)
        EN_to_CPcb = np.dot(EN_to_CPot, CPot_to_CPcb)

        # ---------- Pred 3D ----------
        obj_pts = objpoints[idx].reshape(-1, 3)
        pred_pts = (EN_to_CPcb[:3, :3] @ obj_pts.T + EN_to_CPcb[:3, 3:4]).T  # (N,3)

        # ---------- GT 3D via PnP ----------
        _, Rvec, Tvec = cv2.solvePnP(objpoints[idx], imgpoints[idx], K, D)
        Rvec, _ = cv2.Rodrigues(Rvec)
        Tvec = Tvec.reshape(3)
        gt_pts = (Rvec @ obj_pts.T + Tvec.reshape(3, 1)).T  # (N,3)

        # ---------- RMSE ----------
        err = gt_pts - pred_pts
        rmse_3d = np.sqrt(np.mean(np.sum(err**2, axis=1)))
        all_rmse_3d.append(rmse_3d)

        # ---------- Plot ----------
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], c='b', marker='o', label='GT')
            ax.scatter(pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2], c='r', marker='^', label='Pred')
            for g, p in zip(gt_pts, pred_pts):
                ax.plot([g[0], p[0]], [g[1], p[1]], [g[2], p[2]], 'k--', linewidth=0.5)

            ax.set_title(f"Frame {frame} - RMSE_3D = {rmse_3d:.3f} mm")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_zlabel("Z (mm)")
            ax.legend()

        idx += 1

    return all_rmse_3d

#######################################################################################################
def rmse_3D_disp_plot(polaris_readings, objpoints, imgpoints_L, imgpoints_R, img_dir, frame_names, CPcb_to_CPot, BF_to_EN, stereo_params, plot=True):
    """
    Compute 3D RMSE where GT 3D positions are reconstructed from stereo disparity,
    and predictions come from the EN_to_CPcb chain. Optionally visualize.

    Args:
        polaris_readings (pd.DataFrame): Tracking table with OT poses.
        objpoints (list[np.ndarray]): Per-frame 3D chessboard points (N_i,3).
        imgpoints_L (list[np.ndarray]): Per-frame left 2D detections (N_i,1,2).
        imgpoints_R (list[np.ndarray]): Per-frame right 2D detections (N_i,1,2).
        img_dir (str): Directory containing raw images (not required by the metric).
        frame_names (list[str]): Filenames/IDs to iterate in order.
        CPcb_to_CPot (np.ndarray): 4x4 transform CPcb<-CPot.
        BF_to_EN (np.ndarray): 4x4 transform EN<-BF.
        stereo_params (dict): Dictionary from stereo_calibration (uses 'P_L' and 'T').
        plot (bool): If True, show 3D scatter and per-point error segments.

    Returns:
        all_rmse_3d (list[float]): Per-frame 3D RMSE in mm.
    """
    idx = 0
    all_rmse_3d = []

    for frame in frame_names:
        # Filter tracking rows
        temp = polaris_readings[polaris_readings[' frame name'] == frame]

        OT_to_BF_raw = temp[temp['Tool Type'] == 'E']
        OT_to_CPot_raw = temp[temp['Tool Type'] == 'C']

        # Convert poses to transforms
        OT_to_BF_qt = OT_to_BF_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]
        OT_to_CPot_qt = OT_to_CPot_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]
        OT_to_BF = quaternion_to_transformation(OT_to_BF_qt)
        OT_to_CPot = quaternion_to_transformation(OT_to_CPot_qt)

        # Chain to get EN_to_CPcb
        BF_to_OT = np.linalg.inv(OT_to_BF)
        BF_to_CPot = np.dot(BF_to_OT, OT_to_CPot)
        EN_to_BF = np.linalg.inv(BF_to_EN)
        EN_to_CPot = np.dot(EN_to_BF, BF_to_CPot)
        CPot_to_CPcb = np.linalg.inv(CPcb_to_CPot)
        EN_to_CPcb = np.dot(EN_to_CPot, CPot_to_CPcb)

        # ---------- Predicted 3D points (in EN frame) ----------
        obj_pts = objpoints[idx].reshape(-1, 3)  # chessboard model points
        pred_pts = (EN_to_CPcb[:3, :3] @ obj_pts.T + EN_to_CPcb[:3, 3:4]).T  # (N,3)

        # ---------- Ground truth 3D from disparity ----------
        K = stereo_params['P_L']     # left projection matrix after rectification
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Baseline magnitude from stereo extrinsics (OpenCV convention sign handled)
        B = -stereo_params['T'][0]   # in mm

        # Disparity per corner (u_L - u_R) assuming rectified, row-aligned
        disparity = imgpoints_L[idx][:, 0, 0] - imgpoints_R[idx][:, 0, 0]  # (N,)

        # Depth from disparity (avoid zero disparity in real usage)
        Z = fx * B / disparity
        X = (imgpoints_L[idx][:, 0, 0] - cx) * Z / fx
        Y = (imgpoints_L[idx][:, 0, 1] - cy) * Z / fy
        gt_pts = np.stack([X, Y, Z], axis=1)  # (N,3)

        # ---------- 3D RMSE ----------
        err = gt_pts - pred_pts
        rmse_3d = np.sqrt(np.mean(np.sum(err**2, axis=1)))
        all_rmse_3d.append(rmse_3d)

        # ---------- Optional visualization ----------
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], c='b', marker='o', label='GT')
            ax.scatter(pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2], c='r', marker='^', label='Pred')
            for g, p in zip(gt_pts, pred_pts):
                ax.plot([g[0], p[0]], [g[1], p[1]], [g[2], p[2]], 'k--', linewidth=0.5)

            ax.set_title(f"Frame {frame} - RMSE_3D = {rmse_3d:.3f} mm")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_zlabel("Z (mm)")
            ax.legend()

        idx += 1

    return all_rmse_3d