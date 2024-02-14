import numpy as np
from pyquaternion import Quaternion
from scipy.spatial import ckdtree


def rotation_2d_around_center(pt, center, theta):
    """
    2d rotation on 3d vectors by ignoring y factor
    :param pt:
    :param center:
    :return:
    """
    pt = pt.copy()
    pt = pt - center
    x, y, z = pt
    new_pt = np.array([np.cos(theta) * x - np.sin(theta) * z, y, np.sin(theta) * x + np.cos(theta) * z]) + center
    return new_pt


def extend_along_center(pt, center, add_dist, min_dist, max_dist):
    pt = pt.copy()
    curr_dist = np.linalg.norm(pt - center)
    pt = pt - center
    new_dist = min(max(min_dist, curr_dist + add_dist), max_dist)
    pt = pt * (new_dist / curr_dist)
    pt = pt + center
    return pt


def vectorized_range(start, end):
    """  Return an array of NxD, iterating from the start to the end"""
    N = int(np.max(end - start)) + 1
    idxes = np.floor(np.arange(N) * (end - start)[:, None] / N + start[:, None]).astype('int')
    return idxes


def vectorized_meshgrid(vec_x, vec_y):
    """vec_x in NxK, vec_y in NxD. Return xx in Nx(KxD) and yy in Nx(DxK)"""
    N, K, D = vec_x.shape[0], vec_x.shape[1], vec_y.shape[1]
    vec_x = np.tile(vec_x[:, None, :], [1, D, 1]).reshape(N, -1)
    vec_y = np.tile(vec_y[:, :, None], [1, 1, K]).reshape(N, -1)
    return vec_x, vec_y


def rotate_rigid_object(center, axis, angle, pos=None, relative=None):
    '''
    rotate a rigid object (e.g. shape in flex).

    pos: np.ndarray 3x1, [x, y, z] coordinate of the object.
    relative: relative coordinate of the object to center.
    center: rotation center.
    axis: rotation axis.
    angle: rotation angle in radius.
    TODO: add rotaion of coordinates
    '''

    if relative is None:
        relative = pos - center

    quat = Quaternion(axis=axis, angle=angle)
    after_rotate = quat.rotate(relative)
    return after_rotate + center


def quatFromAxisAngle(axis, angle):
    '''
    given a rotation axis and angle, return a quatirian that represents such roatation.
    '''
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat

def quads2tris(F):
    out = []
    for f in F:
        if len(f) == 3: out += [f]
        elif len(f) == 4: out += [[f[0],f[1],f[2]],
                                [f[0],f[2],f[3]]]
        else: print("This should not happen....")
    return np.array(out, np.int32)

def load_cloth(path):
    """Load .obj of cloth mesh. Only quad-mesh is acceptable!
    Return:
        - vertices: ndarray, (N, 3)
        - triangle_faces: ndarray, (S, 3)
        - stretch_edges: ndarray, (M1, 2)
        - bend_edges: ndarray, (M2, 2)
        - shear_edges: ndarray, (M3, 2)
    This function was written by Zhenjia Xu
    email: xuzhenjia [at] cs (dot) columbia (dot) edu
    website: https://www.zhenjiaxu.com/
    """
    vertices, faces = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # 3D vertex
        if line.startswith('v '):
            vertices.append([float(n)
                             for n in line.replace('v ', '').split(' ')])
        # Face
        elif line.startswith('f '):
            idx = [n.split('/') for n in line.replace('f ', '').split(' ')]
            face = [int(n[0]) - 1 for n in idx]
            assert(len(face) == 4)
            faces.append(face)

    triangle_faces = []
    for face in faces:
        triangle_faces.append([face[0], face[1], face[2]])
        triangle_faces.append([face[0], face[2], face[3]])

    stretch_edges, shear_edges, bend_edges = set(), set(), set()

    # Stretch & Shear
    for face in faces:
        stretch_edges.add(tuple(sorted([face[0], face[1]])))
        stretch_edges.add(tuple(sorted([face[1], face[2]])))
        stretch_edges.add(tuple(sorted([face[2], face[3]])))
        stretch_edges.add(tuple(sorted([face[3], face[0]])))

        shear_edges.add(tuple(sorted([face[0], face[2]])))
        shear_edges.add(tuple(sorted([face[1], face[3]])))

    # Bend
    neighbours = dict()
    for vid in range(len(vertices)):
        neighbours[vid] = set()
    for edge in stretch_edges:
        neighbours[edge[0]].add(edge[1])
        neighbours[edge[1]].add(edge[0])
    for vid in range(len(vertices)):
        neighbour_list = list(neighbours[vid])
        N = len(neighbour_list)
        for i in range(N - 1):
            for j in range(i+1, N):
                bend_edge = tuple(
                    sorted([neighbour_list[i], neighbour_list[j]]))
                if bend_edge not in shear_edges:
                    bend_edges.add(bend_edge)

    return np.array(vertices), np.array(triangle_faces),\
        np.array(list(stretch_edges)), np.array(
            list(bend_edges)), np.array(list(shear_edges))

def find_2D_rigid_trans(pts1, pts2, use_chamfer=False):
    """Given two set aligned points, find the best-fit 2D rigid transformation by kabsch algorithm"""
    uv1 = pts1[:, [0,2]]
    uv2 = pts2[:, [0,2]]
    centroid1 = uv1.mean(0, keepdims=True)
    centroid2 = uv2.mean(0, keepdims=True)
    zero_pts1 = uv1 - centroid1
    zero_pts2 = uv2 - centroid2
    H = zero_pts1.T @ zero_pts2

    U, S, Vh = np.linalg.svd(H)
    d = np.linalg.det(Vh.T @ U.T)
    m = np.eye(2)
    m[1, 1] = d

    R = Vh.T @ m @ U.T
    t = centroid2.T - R@centroid1.T

    rigid_mat = np.concatenate([R, np.zeros((2,1))], axis=1)
    rigid_mat[:, 2:3] += t
    return rigid_mat

def get_closest_point(pred_points, gt_points):
    # pred_tree = ckdtree.cKDTree(pred_points)
    gt_tree = ckdtree.cKDTree(gt_points)
    forward_distance, forward_nn_idx = gt_tree.query(pred_points, k=1)
    # backward_distance, backward_nn_idx = pred_tree.query(gt_points, k=1)
    # forward_chamfer = np.mean(forward_distance)
    # backward_chamfer = np.mean(backward_distance)
    # symmetrical_chamfer = np.mean([forward_chamfer, backward_chamfer])

    return forward_nn_idx
