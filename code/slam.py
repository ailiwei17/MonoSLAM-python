import cv2
import numpy as np
import mapping
import Frame

if __name__ == "__main__":
    "主函数"
    W, H = 960, 540
    F = 270
    K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
    mapp = mapping.Map(1024, 768)
    cap = cv2.VideoCapture("road.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        frame = Frame.Frame(frame, K)

        if ret:
            Frame.Frame.last_kps, Frame.Frame.last_des, Frame.Frame.last_pose = frame.process_frame(mapp)
        else:
            break

        cv2.imshow("slam", frame.image)
        mapp.display()
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
