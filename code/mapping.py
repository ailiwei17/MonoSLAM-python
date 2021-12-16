import numpy as np
import pangolin
import OpenGL.GL as gl
from multiprocessing import Process, Queue

# 构建地图，显示角点的点云和相机的位姿
class Map:
    def __init__(self, W, H):
        self.width = W
        self.Height = H
        self.poses = []
        self.points = []
        self.colors = []
        self.state = None
        self.q = Queue()

        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()

    def add_observation(self, pose, points):
        self.poses.append(pose)
        for point in points:
            self.points.append(point)

    def viewer_init(self):
        pangolin.CreateWindowAndBind('Main', self.width, self.Height)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(self.width, self.Height, 420, 420, self.width//2, self.Height//2, 0.2, 1000),
            pangolin.ModelViewLookAt(0, -10, -8,
                                     0,   0,  0,
                                     0,  -1,  0))
        self.handler = pangolin.Handler3D(self.scam)
        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -self.width/self.Height)
        self.dcam.SetHandler(self.handler)

    def viewer_thread(self, q):
        self.viewer_init()
        while True:
            self.viewer_refresh(q)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        # draw poses
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0])

        # draw keypoints
        gl.glPointSize(2)
        for i, point in enumerate(self.state[1]):
            if point[3] == 1:
                gl.glColor3f(1.0, 0.0, 1.0)
            else:
                color_pb = point[3]
                b = int(color_pb/65536/ 255)
                g = int(color_pb // 65536 / 256 / 255)
                r = int(color_pb // 65536 // 256 / 255)
                gl.glColor3f(r, g, b)
            pangolin.DrawPoints(self.state[1][i:i+1])
        pangolin.FinishFrame()

    def display(self):
        poses = np.array(self.poses)
        points = np.array(self.points)
        self.q.put((poses, points))
