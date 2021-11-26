import cv2
import pangolin_python
import numpy as np
import OpenGL.GL as gl

from multiprocessing import Process, Queue
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform


class Frame(object):
    idx = 0
    last_kps, last_des, last_pose = None, None, None

    def __init__(self, image):
        """把上一帧的信息传递给下一帧"""
        Frame.idx += 1

        self.image = image
        self.idx = Frame.idx
        self.now_kps = Frame.last_kps
        self.now_des = Frame.last_des
        self.now_pose = Frame.last_pose
        self.norm_now_kps = None
        self.norm_last_kps = None

    def extract_points(self):
        """提取角点"""
        orb = cv2.ORB_create()
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(self.image, 3000, qualityLevel=0.01, minDistance=3)
        kps = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=20) for pt in pts]
        kps, des = orb.compute(self.image, kps)
        kps = np.array([(kp.pt[0], kp.pt[1]) for kp in kps])
        return kps, des

    def match_points(self):
        bfmatch = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # 返回k个最佳匹配
        matches = bfmatch.knnMatch(self.now_des, self.last_des, k=2)
        match_kps, idx1, idx2 = [], [], []
        for m, n in matches:
            # 将最佳匹配与次好匹配作比较
            if m.distance < 0.75 * n.distance:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                p1 = self.now_kps[m.queryIdx]
                p2 = self.last_kps[m.trainIdx]

                match_kps.append((p1, p2))
        # 丢失匹配
        assert len(match_kps) >= 8

        self.now_kps = self.now_kps[idx1]
        self.last_kps = self.last_kps[idx2]
        return match_kps

    def fit_essential_matrix(self, match_kps):
        """使用随机采样一致进行去噪"""
        global K
        match_kps = np.array(match_kps)

        # 使用相机内参对角点坐标归一化
        self.norm_now_kps = normalize(K, match_kps[:, 0])
        self.norm_last_kps = normalize(K, match_kps[:, 1])

        # 求解本质矩阵和内点数据
        model, inliers = ransac((self.norm_last_kps, self.norm_now_kps),
                                EssentialMatrixTransform,
                                min_samples=8,
                                residual_threshold=0.005,
                                max_trials=200)

        self.now_kps = self.now_kps[inliers]
        self.last_kps = self.last_kps[inliers]
        return model.params

    def process_frame(self):
        """处理图像"""
        self.now_kps, self.now_des = Frame.extract_points(self)
        Frame.last_kps, Frame.last_des = self.now_kps, self.now_des
        if self.idx == 1:
            self.now_pose = np.eye(4)
            point4d = [[0, 0, 0, 1]]
        else:
            match_kps = Frame.match_points(self)
            # 拟合本质矩阵
            essential_matrix = Frame.fit_essential_matrix(self, match_kps)
            print("---------------- Essential Matrix ----------------")
            print(essential_matrix)
            # 利用本质矩阵分解出相机的位姿
            _, R, t, _ = cv2.recoverPose(essential_matrix, self.norm_now_kps, self.norm_last_kps)


def normalize(K, pts):
    Kinv = np.linalg.inv(K)
    add_ones = lambda x: np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    # 归一化公式
    """
    pts:角点坐标 
    .T:简单转置
    [0:2]:恢复坐标 
    """
    norm_pts = np.dot(Kinv, add_ones(pts).T).T[:, 0:2]
    return norm_pts





if __name__ == "__main__":
    "主函数"
    W, H = 960, 540
    F = 270
    K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = Frame(frame)
        frame.process_frame()
