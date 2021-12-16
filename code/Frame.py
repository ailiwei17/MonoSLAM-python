import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from yoloDetect import detect

class Frame(object):
    idx = 0
    last_kps, last_des, last_pose, last_Rt = None, None, None, np.eye(4)

    def __init__(self, image, K):
        """把上一帧的信息传递给下一帧"""
        Frame.idx += 1
        self.image = image
        self.idx = Frame.idx
        self.last_kps = Frame.last_kps
        self.last_des = Frame.last_des
        self.last_pose = Frame.last_pose
        self.last_Rt = Frame.last_Rt
        self.now_kps, self.now_des, self.now_pose, self.now_Rt = None, None, None, np.eye(4)
        self.norm_now_kps, self.norm_last_kps = None, None
        self.K = K
        self.yolo = True  # 是否打开yolo

    def extract_points(self):
        """提取角点"""
        orb = cv2.ORB_create()
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(image, 3000, qualityLevel=0.01, minDistance=0.1)
        kps = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=20) for pt in pts]
        kps, des = orb.compute(image, kps)
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

    def triangulate(self):
        pose1 = np.linalg.inv(self.last_pose)  # 从世界坐标系变换到相机坐标系的位姿, 因此取逆
        pose2 = np.linalg.inv(self.now_pose)

        pts1 = normalize(self.K, self.last_kps)  # 使用相机内参对角点坐标归一化
        pts2 = normalize(self.K, self.now_kps)

        points4d = np.zeros((pts1.shape[0], 4))
        for i, (kp1, kp2) in enumerate(zip(pts1, pts2)):
            A = np.zeros((4, 4))
            # 角点和相机位姿带入方程
            A[0] = kp1[0] * pose1[2] - pose1[0]
            A[1] = kp1[1] * pose1[2] - pose1[1]
            A[2] = kp2[0] * pose2[2] - pose2[0]
            A[3] = kp2[1] * pose2[2] - pose2[1]
            _, _, vt = np.linalg.svd(A)  # 对 A 进行奇异值分解
            points4d[i] = vt[3]  # x=(u,v,1)

        points4d /= points4d[:, 3:]  # 归一化变换成齐次坐标 [x, y, z, 1]
        return points4d

    def fit_essential_matrix(self, match_kps):
        match_kps = np.array(match_kps)

        # 使用相机内参对角点坐标归一化
        self.norm_now_kps = normalize(self.K, match_kps[:, 0])
        self.norm_last_kps = normalize(self.K, match_kps[:, 1])

        # 求解本质矩阵和内点数据
        model, inliers = ransac((self.norm_last_kps, self.norm_now_kps),
                                EssentialMatrixTransform,
                                min_samples=8,
                                residual_threshold=0.005,
                                max_trials=200)

        self.now_kps = self.now_kps[inliers]
        self.last_kps = self.last_kps[inliers]

        return model.params

    def draw_points(self):
        for kp1, kp2 in zip(self.now_kps, self.last_kps):
            u1, v1 = int(kp1[0]), int(kp1[1])
            u2, v2 = int(kp2[0]), int(kp2[1])
            cv2.circle(self.image, (u1, v1), color=(0, 0, 255), radius=3)
            cv2.line(self.image, (u1, v1), (u2, v2), color=(255, 0, 0))
        return None

    def process_frame(self, mapp):
        """处理图像"""
        self.now_kps, self.now_des = Frame.extract_points(self)
        Frame.last_kps, Frame.last_des, Frame.last_Rt = self.now_kps, self.now_des, self.now_Rt
        if self.idx == 1:
            self.now_pose = np.eye(4)
            points4d = [[0, 0, 0, 1]]
        else:
            match_kps = Frame.match_points(self)
            # 拟合本质矩阵
            essential_matrix = Frame.fit_essential_matrix(self, match_kps)
            print("---------------- Essential Matrix ----------------")
            print(essential_matrix)
            # 利用本质矩阵分解出相机的位姿
            _, R, t, _ = cv2.recoverPose(essential_matrix, self.norm_now_kps, self.norm_last_kps)
            self.now_Rt[:3, :3] = R
            self.now_Rt[:3, 3] = t.flatten()
            # 计算当前帧相当于上一帧的位姿变化
            self.now_pose = np.dot(self.now_Rt, self.last_pose)
            # 三角测量得到空间位置
            points4d = Frame.triangulate(self)
            good_pt4d = check_points(points4d)
            points4d = points4d[good_pt4d]
            self.last_kps = self.last_kps[good_pt4d]
            self.now_kps = self.now_kps[good_pt4d]
            if self.yolo:
                points4d = points4d.tolist()
                img, boxes, colors = detect(self.image)
                colors.extend(colors[0:(len(boxes) > len(colors))]) if len(boxes) != len(colors) else 0
                if boxes != 0:
                    for i in range(len(boxes[0])):
                        (x, y) = (boxes[i][0] / 608 * self.image.shape[1], boxes[i][1] / 608 * self.image.shape[0])
                        (w, h) = (boxes[i][2] / 608 * self.image.shape[1], boxes[i][3] / 608 * self.image.shape[0])
                        for (j, point4d) in enumerate(points4d):
                            if (x - h < point4d[0] < x + h) & (y - w < point4d[1] < y + w):
                                point4d[3] = 65536 * colors[i][2] + 256 * colors[i][1] + colors[i][0]
                img = cv2.resize(img, (self.image.shape[1], self.image.shape[0]))
                cv2.imshow("test", img)
            Frame.draw_points(self)

        mapp.add_observation(self.now_pose, points4d)  # 将当前的 pose 和点云放入地图中
        # 将当前帧的pose传递给下一帧
        Frame.last_pose = self.now_pose
        Frame.last_Rt = self.now_Rt
        return self

def check_points(points4d):
    # 判断3D点是否在两个摄像头前方
    good_points = points4d[:, 2] > 0
    # TODO: parallax、BA等等 ....
    return good_points


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
