import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
############################################################################################################
def find_corners(left_dir, right_dir, left_img_list, right_img_list, objp, chessboard_size):
    objpoints = []  # 3D points in real world space
    imgpoints_L = []  # left 2D
    imgpoints_R = [] # right 2D

    num_success = 0
    success_frames = []
    num_fail = 0

    for img_name_L, img_name_R in zip(left_img_list, right_img_list):
        # Image Paths
        path_L = os.path.join(left_dir, img_name_L)
        path_R = os.path.join(right_dir, img_name_R)

        img_L = cv2.imread(path_L)
        img_R = cv2.imread(path_R)

        gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY

        ret_L, corners_L = cv2.findChessboardCornersSB(gray_L, chessboard_size, flags)
        ret_R, corners_R = cv2.findChessboardCornersSB(gray_R, chessboard_size, flags)

        if ret_L and ret_R:
            imgpoints_L.append(corners_L)
            imgpoints_R.append(corners_R)
            objpoints.append(objp)

            num_success += 1
            success_frames.append([img_name_L, img_name_R])
        else:
            num_fail += 1

    cv2.destroyAllWindows()

    return imgpoints_L, imgpoints_R, objpoints, success_frames

#############################################################################################################
import cv2
import numpy as np

def stereo_calibration(objpoints, imgpoints_L, imgpoints_R, img_size,
                             K_L=None, D_L=None, K_R=None, D_R=None,
                             criteria=None, flags=0):

    # Stereo calibration (joint)
    ret, K_L, D_L, K_R, D_R, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_L, imgpoints_R,
        K_L, D_L, K_R, D_R,
        img_size,
        criteria=criteria,
        flags=flags
    )

    # Stereo rectification
    R_L, R_R, P_L, P_R, Q, ROI1, ROI2 = cv2.stereoRectify(
        K_L, D_L, K_R, D_R, img_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
    )

    # Pack results
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
    # Calculate 2D Reprojection Errors
    all_rmse_px = []
    all_rmse_mm = []
    dist = []
    for i in range(len(objpoints)):
        ############### Left ###########################
        _, Rvec, Tvec = cv2.solvePnP(objpoints[i], imgpoints[i], K, D) # Transformation between Left-Eye and Detected Corners
        proj, _ = cv2.projectPoints(objpoints[i], Rvec, Tvec, K, D) # Reproject Points onto Left Image (2D)

        ####### pixel RMSE ############
        real_px = np.asarray(imgpoints[i], dtype=np.float32)
        pred_px = np.asarray(proj, dtype=np.float32)

        err_px = real_px - pred_px
        err_px = err_px.reshape(-1, 2)  # (N, 2)

        rmse_px = np.sqrt(np.mean(err_px[:,0]**2 + err_px[:,1]**2))
        all_rmse_px.append(rmse_px)

        ######## mm RMSE ################
        fx, fy = K[0, 0], K[1, 1]

        Rot, _ = cv2.Rodrigues(Rvec)
        Tran = Tvec
        obj_pts = objpoints[i].reshape(-1, 3)  # shape (N, 3)
        pts_cam = (Rot @ obj_pts.T + Tran).T     # Real Ground Truth Corner Positions in 3D, shape (N, 3)
        Z_vals = pts_cam[:, 2]                # z-distance of each point (mm)

        err_mm_x = (err_px[:, 0] * Z_vals / fx)
        err_mm_y = (err_px[:, 1] * Z_vals / fy)

        rmse_mm = np.sqrt(np.mean(err_mm_x**2 + err_mm_y**2))
        all_rmse_mm.append(rmse_mm)
        dist.append(float(Tvec[2]))

    return all_rmse_px, all_rmse_mm, dist

#####################################################################################################
def plot_rmse_dist(dist, rmse, frame_names, unit, side):
    # Visualizing RMSEs
    from scipy.stats import pearsonr
    import matplotlib.pyplot as plt

    # Pearson correlation
    corr, pval= pearsonr(dist, rmse)
    plt.figure(figsize=(10, 4))

    plt.scatter(dist, rmse)

    # Labeling each point with frame names
    for x, y, name in zip(dist, rmse, frame_names):
        plt.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),  
            fontsize=8
        )

    plt.xlabel('Working Distance (mm)')
    plt.ylabel('Reprojection RMSE '+ unit)
    plt.title(side + f'Camera\nCorrelation: {corr:.3f}' + f'p-value:{pval:.3f}')
    plt.grid(True)


##############################################################################################
def undistort_rectify_save(img, save_path, map1, map2):
    img_rect = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(save_path, img_rect)

##############################################################################################
def quaternion_to_transformation(quaternion):
    from scipy.spatial.transform import Rotation as R
    q0, qx, qy, qz = float(quaternion['q0']), float(quaternion['qx']), float(quaternion['qy']), float(quaternion['qz'])
    tx, ty, tz =  float(quaternion['tx']), float(quaternion['ty']), float(quaternion['tz'])

    rotation = R.from_quat([q0, qx, qy, qz], scalar_first=True)
    rotation_matrix = rotation.as_matrix()  # 3x3 旋转矩阵

    # 构建 4x4 齐次变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]

    return transformation_matrix

def quaternion_to_transformation_Mike(quaternion):
    q0, qx, qy, qz = float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3])
    tx, ty, tz =  float(quaternion[4]), float(quaternion[5]), float(quaternion[6])

    rotation = R.from_quat([q0, qx, qy, qz], scalar_first=True)
    rotation_matrix = rotation.as_matrix()  # 3x3 旋转矩阵

    # 构建 4x4 齐次变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [tx, ty, tz]

    return transformation_matrix

######################################################################################################
def calculate_BF_to_EN(polaris_readings, objpoints, imgpoints, frame_names, CPcb_to_CPot, K, D):
    BF_to_EN_list = []

    EN_to_CPcb_list = []

    CPcb_to_EN_list = []

    BF_to_CPcb_list = []

    idx = 0
    for frame in frame_names:
        temp = polaris_readings[polaris_readings[' frame name'] == frame]
        # E is for endoscope (PO_to_BF)
        # C is for calibration plate (PO_to_CBrom)
        OT_to_BF_raw = temp[temp['Tool Type'] == 'E']
        OT_to_CPot_raw = temp[temp['Tool Type'] == 'C']
        # get the quaternions
        OT_to_BF_qt = OT_to_BF_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]
        OT_to_CPot_qt = OT_to_CPot_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]

        # Convert quaternions to Transformation matrices
        OT_to_BF = quaternion_to_transformation(OT_to_BF_qt)
        OT_to_CPot = quaternion_to_transformation(OT_to_CPot_qt)
        # Transformation Chain
        BF_to_OT = np.linalg.inv(OT_to_BF)
        BF_to_CPot = np.dot(BF_to_OT, OT_to_CPot)
        CPot_to_CPcb = np.linalg.inv(CPcb_to_CPot)
        BF_to_CPcb = np.dot(BF_to_CPot, CPot_to_CPcb)
        BF_to_CPcb_list.append(BF_to_CPcb)
        # load EN_to_CPcb from Rvec and Tvec
        _, Rvec, Tvec = cv2.solvePnP(objpoints[idx], imgpoints[idx], K, D) # Transformation between Left-Eye and Detected Corners
        temp_Rot = Rvec
        temp_tran = Tvec
        temp_Rot, _ = cv2.Rodrigues(temp_Rot)
        
        EN_to_CPcb = np.eye(4)
        EN_to_CPcb[:3, :3] = temp_Rot
        EN_to_CPcb[:3, 3] = temp_tran.flatten()
        EN_to_CPcb_list.append(EN_to_CPcb)

        CPcb_to_EN = np.linalg.inv(EN_to_CPcb)
        CPcb_to_EN_list.append(CPcb_to_EN)
        BF_to_EN = np.dot(BF_to_CPcb, CPcb_to_EN)

        BF_to_EN_list.append(BF_to_EN)

        idx = idx + 1
    
    return BF_to_EN_list, CPcb_to_EN_list, EN_to_CPcb_list, BF_to_CPcb_list

#######################################################################################################
from scipy.spatial.transform import Rotation as R

def average_rotation_matrix_svd(R_list):
    """使用 SVD 对多个旋转矩阵求平均"""
    M = np.zeros((3, 3))
    for R_i in R_list:
        M += R_i
    U, _, Vt = np.linalg.svd(M)
    R_avg = U @ Vt
    # 修正为合法旋转（det > 0）
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return R_avg

def estimate_transform_average(B_list, C_list):
    """
    给定 B_i 和 C_i，估计 A ≈ B_i * C_i 的最优平均值（刚体变换）
    
    参数：
        B_list: List[np.ndarray], 每个是 4x4 变换矩阵
        C_list: List[np.ndarray], 每个是 4x4 变换矩阵
    
    返回：
        A_avg: 4x4 np.ndarray，估计出的平均变换
    """
    assert len(B_list) == len(C_list), "B_list 和 C_list 长度不一致"
    N = len(B_list)
    A_list = [B_list[i] @ C_list[i] for i in range(N)]

    # 平移平均
    t_list = np.array([A[:3, 3] for A in A_list])
    t_avg = np.mean(t_list, axis=0)

    # 旋转平均（用 SVD）
    R_list = [A[:3, :3] for A in A_list]
    R_avg = average_rotation_matrix_svd(R_list)

    # 构造最终 A
    A_avg = np.eye(4)
    A_avg[:3, :3] = R_avg
    A_avg[:3, 3] = t_avg
    return A_avg

#######################################################################################################
def rmse_2D_transform(polaris_readings, objp, objpoints, imgpoints, chessboard_size, img_dir, frame_names, CPcb_to_CPot, BF_to_EN, K, D, save_img = False, save_dir=None):
    idx = 0
    all_rmse_px = []
    all_rmse_mm = []
    dist = []
    for frame in frame_names:
        # read rows from .csv based on frame name
        temp = polaris_readings[polaris_readings[' frame name'] == frame]
        img = cv2.imread(os.path.join(img_dir, frame))
        # E is for endoscope (PO_to_BF)
        # C is for calibration plate (PO_to_CBrom)
        OT_to_BF_raw = temp[temp['Tool Type'] == 'E']
        OT_to_CPot_raw = temp[temp['Tool Type'] == 'C']

        # get the quaternions
        OT_to_BF_qt = OT_to_BF_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]
        OT_to_CPot_qt = OT_to_CPot_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]

        # Convert quaternions to Transformation matrices
        OT_to_BF = quaternion_to_transformation(OT_to_BF_qt)
        OT_to_CPot = quaternion_to_transformation(OT_to_CPot_qt)

        # Transformation Chain
        BF_to_OT = np.linalg.inv(OT_to_BF)
        BF_to_CPot = np.dot(BF_to_OT, OT_to_CPot)

        EN_to_BF = np.linalg.inv(BF_to_EN)
        EN_to_CPot = np.dot(EN_to_BF, BF_to_CPot)
        CPot_to_CPcb = np.linalg.inv(CPcb_to_CPot)
        EN_to_CPcb = np.dot(EN_to_CPot, CPot_to_CPcb)

        Rot_EN_to_CPcb = EN_to_CPcb[:3, :3]
        Tran_EN_to_CPcb = EN_to_CPcb[:3, 3]

        Rot, _ = cv2.Rodrigues(Rot_EN_to_CPcb)
        Tran = Tran_EN_to_CPcb
        projected_points, _ = cv2.projectPoints(objp, Rot, Tran, K, D)        

        if save_img:
            cv2.drawChessboardCorners(img, chessboard_size, projected_points, True)
            save_path = os.path.join(save_dir, frame)
            cv2.imwrite(save_path, img) 
        
        ###### pixel RMSE ############
        real_pts = np.asarray(imgpoints[idx], dtype=np.float32)
        pred_pts = np.asarray(projected_points, dtype=np.float32)

        err_px = real_pts - pred_pts
        err_px = err_px.reshape(-1, 2)  # (N, 2)

        rmse_px = np.sqrt(np.mean(np.sum(err_px**2, axis=1)))
        all_rmse_px.append(rmse_px)
        ######## mm RMSE ################
        fx = K[0, 0]
        fy = K[1, 1]

        _, Rvec, Tvec = cv2.solvePnP(objpoints[idx], imgpoints[idx], K, D) # Transformation between Left-Eye and Detected Corners
        Rot, _ = cv2.Rodrigues(Rvec)
        Tran = Tvec

        obj_pts = objpoints[idx].reshape(-1, 3)  # shape (N, 3)
        pts_cam = (Rot @ obj_pts.T + Tran).T       # shape (N, 3)
        Z_vals = pts_cam[:, 2]                # 每个点的深度 (单位：mm)

        err_mm_x = (err_px[:, 0] * Z_vals / fx)
        err_mm_y = (err_px[:, 1] * Z_vals / fy)

        rmse_mm = np.sqrt(np.mean(err_mm_x**2 + err_mm_y**2))
        all_rmse_mm.append(rmse_mm)

        dist.append(float(Tvec[2]))

        idx = idx + 1
    
    return all_rmse_px, all_rmse_mm, dist


#######################################################################################################
def rmse_3D_transform(polaris_readings, objpoints, imgpoints, img_dir, frame_names, CPcb_to_CPot, BF_to_EN, K, D):
    idx = 0
    all_rmse_3D = []
    dist = []
    for frame in frame_names:
        # read rows from .csv based on frame name
        temp = polaris_readings[polaris_readings[' frame name'] == frame]
        img = cv2.imread(os.path.join(img_dir, frame))
        # E is for endoscope (PO_to_BF)
        # C is for calibration plate (PO_to_CBrom)
        OT_to_BF_raw = temp[temp['Tool Type'] == 'E']
        OT_to_CPot_raw = temp[temp['Tool Type'] == 'C']

        # get the quaternions
        OT_to_BF_qt = OT_to_BF_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]
        OT_to_CPot_qt = OT_to_CPot_raw[['q0','qx', 'qy', 'qz', 'tx', 'ty', 'tz']]

        # Convert quaternions to Transformation matrices
        OT_to_BF = quaternion_to_transformation(OT_to_BF_qt)
        OT_to_CPot = quaternion_to_transformation(OT_to_CPot_qt)

        # Transformation Chain
        BF_to_OT = np.linalg.inv(OT_to_BF)
        BF_to_CPot = np.dot(BF_to_OT, OT_to_CPot)

        EN_to_BF = np.linalg.inv(BF_to_EN)
        EN_to_CPot = np.dot(EN_to_BF, BF_to_CPot)
        CPot_to_CPcb = np.linalg.inv(CPcb_to_CPot)
        EN_to_CPcb = np.dot(EN_to_CPot, CPot_to_CPcb)

        Rot_EN_to_CPcb = EN_to_CPcb[:3, :3]
        Tran_EN_to_CPcb = EN_to_CPcb[:3, 3]

        obj_pts = objpoints[idx].reshape(-1, 3).astype(np.float64)
        pts_pred = (Rot_EN_to_CPcb @ obj_pts.T + Tran_EN_to_CPcb.reshape(3,1)).T  # (N,3)

        _, Rvec, Tvec = cv2.solvePnP(objpoints[idx], imgpoints[idx], K, D) # Transformation between Left-Eye and Detected Corners
        Rvec, _ = cv2.Rodrigues(Rvec)
        Tvec = Tvec.reshape(3)
        pts_gt = (Rvec @ obj_pts.T + Tvec.reshape(3,1)).T  # (N,3)

        # 3D RMSE
        err = pts_pred - pts_gt              # (N,3)
        rmse_3d = np.sqrt(np.mean(err[:,0]**2 + err[:,1]**2 + err[:,2]**2))
        all_rmse_3D.append(float(rmse_3d))

        # 可选：记录距离（比如GT的Z）
        dist.append(float(Tvec[2]))

        idx += 1
    return all_rmse_3D, dist

#######################################################################################################

def rmse_3D_transform_plot(polaris_readings, objpoints, imgpoints, img_dir, frame_names, CPcb_to_CPot, BF_to_EN, K, D, plot=True):
    idx = 0
    all_rmse_3d = []

    for frame in frame_names:
        # 取对应帧的数据
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
        obj_pts = objpoints[idx].reshape(-1, 3)  # GT in CB coord
        pred_pts = (EN_to_CPcb[:3, :3] @ obj_pts.T + EN_to_CPcb[:3, 3:4]).T  # (N, 3)

        # ---------- Compute 3D RMSE ----------
        _, Rvec, Tvec = cv2.solvePnP(objpoints[idx], imgpoints[idx], K, D) # Transformation between Left-Eye and Detected Corners
        Rvec, _ = cv2.Rodrigues(Rvec)
        Tvec = Tvec.reshape(3)
        gt_pts = (Rvec @ obj_pts.T + Tvec.reshape(3,1)).T  # (N,3)

        err = gt_pts - pred_pts
        rmse_3d = np.sqrt(np.mean(np.sum(err**2, axis=1)))
        all_rmse_3d.append(rmse_3d)

        # ---------- Plot ----------
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(gt_pts[:,0], gt_pts[:,1], gt_pts[:,2], c='b', marker='o', label='GT')
            ax.scatter(pred_pts[:,0], pred_pts[:,1], pred_pts[:,2], c='r', marker='^', label='Pred')
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
    idx = 0
    all_rmse_3d = []

    for frame in frame_names:
        # 取对应帧的数据
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
        obj_pts = objpoints[idx].reshape(-1, 3)  # GT in CB coord
        pred_pts = (EN_to_CPcb[:3, :3] @ obj_pts.T + EN_to_CPcb[:3, 3:4]).T  # (N, 3)
        # ---------- Conmpute Ground truth using Disparity ----------
        K= stereo_params['P_L']
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        B = -stereo_params['T'][0]  # Baseline in mm (negative sign because of OpenCV convention)
        disparity = imgpoints_L[idx][:,0,0] - imgpoints_R[idx][:,0,0]  # (N,)
        Z = fx * B / disparity  # Depth in mm
        X = (imgpoints_L[idx][:,0,0] - cx) * Z / fx
        Y = (imgpoints_L[idx][:,0,1] - cy) * Z / fy
        gt_pts = np.stack([X, Y, Z], axis=1)  # (N, 3)
        # ---------- Calculate 3D RMSE ----------
        err = gt_pts - pred_pts
        rmse_3d = np.sqrt(np.mean(np.sum(err**2, axis=1)))
        all_rmse_3d.append(rmse_3d)

        # ---------- Plot ----------
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(gt_pts[:,0], gt_pts[:,1], gt_pts[:,2], c='b', marker='o', label='GT')
            ax.scatter(pred_pts[:,0], pred_pts[:,1], pred_pts[:,2], c='r', marker='^', label='Pred')
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
######################################################################################################
def calculate_BF_to_EN_Mike(matching, objpoints, imgpoints, frame_names, CPcb_to_CPot, K, D):
    import pandas as pd

    BF_to_EN_list = []

    EN_to_CPcb_list = []

    CPcb_to_EN_list = []

    BF_to_CPcb_list = []

    idx = 0
    for frame in frame_names:
        temp = matching[(matching['tiff_L'] == frame[:-4]) | (matching['tiff_R'] == frame[:-4])]
        polaris_csv = temp['pose_id'].values[0] + '.csv'

        polaris_csv_path = os.path.join("C:\\Users\\f007wsq\\Desktop\\datasets\\20250619-xi-testing", "polaris", polaris_csv)
        polaris_data = pd.read_csv(polaris_csv_path)
        raw = np.array(polaris_data)
        
        # get the quaternions
        OT_to_BF_qt = raw[0][40:47]
        OT_to_CPot_qt = raw[0][70:77]  

        # Convert quaternions to Transformation matrices
        OT_to_BF = quaternion_to_transformation_Mike(OT_to_BF_qt)
        OT_to_CPot = quaternion_to_transformation_Mike(OT_to_CPot_qt)
        # Transformation Chain
        BF_to_OT = np.linalg.inv(OT_to_BF)
        BF_to_CPot = np.dot(BF_to_OT, OT_to_CPot)
        CPot_to_CPcb = np.linalg.inv(CPcb_to_CPot)
        BF_to_CPcb = np.dot(BF_to_CPot, CPot_to_CPcb)
        BF_to_CPcb_list.append(BF_to_CPcb)
        # load EN_to_CPcb from Rvec and Tvec
        _, Rvec, Tvec = cv2.solvePnP(objpoints[idx], imgpoints[idx], K, D) # Transformation between Left-Eye and Detected Corners
        temp_Rot = Rvec
        temp_tran = Tvec
        temp_Rot, _ = cv2.Rodrigues(temp_Rot)
        
        EN_to_CPcb = np.eye(4)
        EN_to_CPcb[:3, :3] = temp_Rot
        EN_to_CPcb[:3, 3] = temp_tran.flatten()
        EN_to_CPcb_list.append(EN_to_CPcb)

        CPcb_to_EN = np.linalg.inv(EN_to_CPcb)
        CPcb_to_EN_list.append(CPcb_to_EN)
        BF_to_EN = np.dot(BF_to_CPcb, CPcb_to_EN)

        BF_to_EN_list.append(BF_to_EN)

        idx = idx + 1
    
    return BF_to_EN_list, CPcb_to_EN_list, EN_to_CPcb_list, BF_to_CPcb_list

#######################################################################################################
def rmse_2D_transform_Mike(matching, objp, objpoints, imgpoints, chessboard_size, img_dir, frame_names, CPcb_to_CPot, BF_to_EN, K, D, save_img = False, save_dir=None):
    idx = 0
    all_rmse_px = []
    all_rmse_mm = []
    dist = []
    for frame in frame_names:
        img = cv2.imread(os.path.join(img_dir, frame))

        temp = matching[(matching['tiff_L'] == frame[:-4]) | (matching['tiff_R'] == frame[:-4])]
        polaris_csv = temp['pose_id'].values[0] + '.csv'

        polaris_csv_path = os.path.join("C:\\Users\\f007wsq\\Desktop\\datasets\\20250619-xi-testing", "polaris", polaris_csv)
        polaris_data = pd.read_csv(polaris_csv_path)
        raw = np.array(polaris_data)
        
        # get the quaternions
        OT_to_BF_qt = raw[0][40:47]
        OT_to_CPot_qt = raw[0][70:77]  

        # Convert quaternions to Transformation matrices
        OT_to_BF = quaternion_to_transformation_Mike(OT_to_BF_qt)
        OT_to_CPot = quaternion_to_transformation_Mike(OT_to_CPot_qt)

        # Transformation Chain
        BF_to_OT = np.linalg.inv(OT_to_BF)
        BF_to_CPot = np.dot(BF_to_OT, OT_to_CPot)

        EN_to_BF = np.linalg.inv(BF_to_EN)
        EN_to_CPot = np.dot(EN_to_BF, BF_to_CPot)
        CPot_to_CPcb = np.linalg.inv(CPcb_to_CPot)
        EN_to_CPcb = np.dot(EN_to_CPot, CPot_to_CPcb)

        Rot_EN_to_CPcb = EN_to_CPcb[:3, :3]
        Tran_EN_to_CPcb = EN_to_CPcb[:3, 3]

        Rot, _ = cv2.Rodrigues(Rot_EN_to_CPcb)
        Tran = Tran_EN_to_CPcb
        projected_points, _ = cv2.projectPoints(objp, Rot, Tran, K, D)        

        if save_img:
            cv2.drawChessboardCorners(img, chessboard_size, projected_points, True)
            save_path = os.path.join(save_dir, frame)
            cv2.imwrite(save_path, img) 
        
        ###### pixel RMSE ############
        real_pts = np.asarray(imgpoints[idx], dtype=np.float32)
        pred_pts = np.asarray(projected_points, dtype=np.float32)

        err_px = real_pts - pred_pts
        err_px = err_px.reshape(-1, 2)  # (N, 2)

        rmse_px = np.sqrt(np.mean(np.sum(err_px**2, axis=1)))
        all_rmse_px.append(rmse_px)
        ######## mm RMSE ################
        fx = K[0, 0]
        fy = K[1, 1]

        _, Rvec, Tvec = cv2.solvePnP(objpoints[idx], imgpoints[idx], K, D) # Transformation between Left-Eye and Detected Corners
        Rot, _ = cv2.Rodrigues(Rvec)
        Tran = Tvec

        obj_pts = objpoints[idx].reshape(-1, 3)  # shape (N, 3)
        pts_cam = (Rot @ obj_pts.T + Tran).T       # shape (N, 3)
        Z_vals = pts_cam[:, 2]                # 每个点的深度 (单位：mm)

        err_mm_x = (err_px[:, 0] * Z_vals / fx)
        err_mm_y = (err_px[:, 1] * Z_vals / fy)

        rmse_mm = np.sqrt(np.mean(err_mm_x**2 + err_mm_y**2))
        all_rmse_mm.append(rmse_mm)

        dist.append(float(Tvec[2]))

        idx = idx + 1
    
    return all_rmse_px, all_rmse_mm, dist
#######################################################################################################
def rmse_3D_transform_plot_Mike(matching, objpoints, imgpoints, img_dir, frame_names, CPcb_to_CPot, BF_to_EN, K, D, plot=True):
    idx = 0
    all_rmse_3d = []

    for frame in frame_names:
        temp = matching[(matching['tiff_L'] == frame[:-4]) | (matching['tiff_R'] == frame[:-4])]
        polaris_csv = temp['pose_id'].values[0] + '.csv'

        polaris_csv_path = os.path.join("C:\\Users\\f007wsq\\Desktop\\datasets\\20250619-xi-testing", "polaris", polaris_csv)
        polaris_data = pd.read_csv(polaris_csv_path)
        raw = np.array(polaris_data)
        
        # get the quaternions
        OT_to_BF_qt = raw[0][40:47]
        OT_to_CPot_qt = raw[0][70:77]  

        # Convert quaternions to Transformation matrices
        OT_to_BF = quaternion_to_transformation_Mike(OT_to_BF_qt)
        OT_to_CPot = quaternion_to_transformation_Mike(OT_to_CPot_qt)

        BF_to_OT = np.linalg.inv(OT_to_BF)
        BF_to_CPot = np.dot(BF_to_OT, OT_to_CPot)

        EN_to_BF = np.linalg.inv(BF_to_EN)
        EN_to_CPot = np.dot(EN_to_BF, BF_to_CPot)
        CPot_to_CPcb = np.linalg.inv(CPcb_to_CPot)
        EN_to_CPcb = np.dot(EN_to_CPot, CPot_to_CPcb)

        # ---------- Pred 3D ----------
        obj_pts = objpoints[idx].reshape(-1, 3)  # GT in CB coord
        pred_pts = (EN_to_CPcb[:3, :3] @ obj_pts.T + EN_to_CPcb[:3, 3:4]).T  # (N, 3)

        # ---------- Compute 3D RMSE ----------
        _, Rvec, Tvec = cv2.solvePnP(objpoints[idx], imgpoints[idx], K, D) # Transformation between Left-Eye and Detected Corners
        Rvec, _ = cv2.Rodrigues(Rvec)
        Tvec = Tvec.reshape(3)
        gt_pts = (Rvec @ obj_pts.T + Tvec.reshape(3,1)).T  # (N,3)

        err = gt_pts - pred_pts
        rmse_3d = np.sqrt(np.mean(np.sum(err**2, axis=1)))
        all_rmse_3d.append(rmse_3d)

        # ---------- Plot ----------
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(gt_pts[:,0], gt_pts[:,1], gt_pts[:,2], c='b', marker='o', label='GT')
            ax.scatter(pred_pts[:,0], pred_pts[:,1], pred_pts[:,2], c='r', marker='^', label='Pred')
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
def rmse_3D_disp_plot_Mike(matching, objpoints, imgpoints_L, imgpoints_R, img_dir, frame_names, CPcb_to_CPot, BF_to_EN, stereo_params, plot=True):
    idx = 0
    all_rmse_3d = []

    for frame in frame_names:
        temp = matching[(matching['tiff_L'] == frame[:-4]) | (matching['tiff_R'] == frame[:-4])]
        polaris_csv = temp['pose_id'].values[0] + '.csv'

        polaris_csv_path = os.path.join("C:\\Users\\f007wsq\\Desktop\\datasets\\20250619-xi-testing", "polaris", polaris_csv)
        polaris_data = pd.read_csv(polaris_csv_path)
        raw = np.array(polaris_data)
        
        # get the quaternions
        OT_to_BF_qt = raw[0][40:47]
        OT_to_CPot_qt = raw[0][70:77]  

        # Convert quaternions to Transformation matrices
        OT_to_BF = quaternion_to_transformation_Mike(OT_to_BF_qt)
        OT_to_CPot = quaternion_to_transformation_Mike(OT_to_CPot_qt)

        BF_to_OT = np.linalg.inv(OT_to_BF)
        BF_to_CPot = np.dot(BF_to_OT, OT_to_CPot)

        EN_to_BF = np.linalg.inv(BF_to_EN)
        EN_to_CPot = np.dot(EN_to_BF, BF_to_CPot)
        CPot_to_CPcb = np.linalg.inv(CPcb_to_CPot)
        EN_to_CPcb = np.dot(EN_to_CPot, CPot_to_CPcb)

        # ---------- Pred 3D ----------
        obj_pts = objpoints[idx].reshape(-1, 3)  # GT in CB coord
        pred_pts = (EN_to_CPcb[:3, :3] @ obj_pts.T + EN_to_CPcb[:3, 3:4]).T  # (N, 3)
        # ---------- Conmpute Ground truth using Disparity ----------
        K= stereo_params['P_L']
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        B = -stereo_params['T'][0]  # Baseline in mm (negative sign because of OpenCV convention)
        disparity = imgpoints_L[idx][:,0,0] - imgpoints_R[idx][:,0,0]  # (N,)
        Z = fx * B / disparity  # Depth in mm
        X = (imgpoints_L[idx][:,0,0] - cx) * Z / fx
        Y = (imgpoints_L[idx][:,0,1] - cy) * Z / fy
        gt_pts = np.stack([X, Y, Z], axis=1)  # (N, 3)
        # ---------- Calculate 3D RMSE ----------
        err = gt_pts - pred_pts
        rmse_3d = np.sqrt(np.mean(np.sum(err**2, axis=1)))
        all_rmse_3d.append(rmse_3d)

        # ---------- Plot ----------
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(gt_pts[:,0], gt_pts[:,1], gt_pts[:,2], c='b', marker='o', label='GT')
            ax.scatter(pred_pts[:,0], pred_pts[:,1], pred_pts[:,2], c='r', marker='^', label='Pred')
            for g, p in zip(gt_pts, pred_pts):
                ax.plot([g[0], p[0]], [g[1], p[1]], [g[2], p[2]], 'k--', linewidth=0.5)

            ax.set_title(f"Frame {frame} - RMSE_3D = {rmse_3d:.3f} mm")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_zlabel("Z (mm)")
            ax.legend()

        idx += 1

    return all_rmse_3d