import g2o
import numpy as np

def optimize(frame, points, K, verbose=False, rounds=200):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(solver)
    robust_kernel = g2o.RobustKernelHuber()

    # add normalized camera
    cam = g2o.CameraParameters(K[0, 0], (K[0, 2], K[1, 2]), 0)
    cam.set_id(0)
    opt.add_parameter(cam)

    last_pose = frame.last_pose
    se3 = g2o.SE3Quat(last_pose[0:3, 0:3], last_pose[0:3, 3])
    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_estimate(se3)
    v_se3.set_id(0)
    opt.add_vertex(v_se3)

    now_pose = frame.now_pose
    se3 = g2o.SE3Quat(now_pose[0:3, 0:3], now_pose[0:3, 3])
    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_estimate(se3)
    v_se3.set_id(1)
    opt.add_vertex(v_se3)

    for (i, kps) in enumerate(frame.last_kps):
        pt = g2o.VertexSBAPointXYZ()
        pt.set_id(i+2)
        z = 1
        x = (kps[0] - cx) * z / fx
        y = (kps[1] - cy) * z / fy
        pt.set_marginalized(True)
        pt.set_estimate([x, y, z])
        opt.add_vertex(pt)

    for (i, kps) in enumerate(frame.last_kps):
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_id(i)
        edge.set_vertex(0, opt.vertex(i+2))
        edge.set_vertex(1, opt.vertex(0))
        edge.set_parameter_id(0, 0)
        edge.set_measurement(kps)
        edge.set_information(np.eye(2))
        edge.set_robust_kernel(robust_kernel)
        opt.add_edge(edge)

    for (i, kps) in enumerate(frame.now_kps):
        edge = g2o.EdgeProjectXYZ2UV()
        edge.set_id(i)
        edge.set_vertex(0, opt.vertex(i+2))
        edge.set_vertex(1, opt.vertex(1))
        edge.set_parameter_id(0, 0)
        edge.set_measurement(kps)
        edge.set_information(np.eye(2))
        edge.set_robust_kernel(robust_kernel)
        opt.add_edge(edge)

    if verbose:
        opt.set_verbose(True)
    opt.initialize_optimization()
    opt.optimize(rounds)

    est = opt.vertex(1).estimate()
    R = est.rotation().matrix()
    t = est.translation()
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret













