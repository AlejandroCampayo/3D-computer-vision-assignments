import os
import json
import math
import numpy as np
import trimesh
import plyfile

def get_pose_json(workspace_path: str, name: str):

    # TODO: only if dense folder exists and count of folder should 0
    image_path = os.path.join("./images")
    intrinsic_path = os.path.join(workspace_path, "dense/0/sparse/cameras.txt")
    extrinsic_path = os.path.join(workspace_path, "dense/0/sparse/images.txt")
    # intrinsic_path = os.path.join(workspace_path, "sparse/0/cameras.txt")
    # extrinsic_path = os.path.join(workspace_path, "sparse/0/images.txt")
    if 'dense' in os.listdir(workspace_path):
        print('using pointcloud from dense reconstruction')
        pointcloud_path = os.path.join(workspace_path, "dense/0/fused.ply")
    else:
        print('using given pointcloud')
        pointcloud_path = os.path.join("./fused.ply")

    out = {}

    intrinsic_mat, w, h = get_intrinsic(intrinsic_path)
    # update out with intrinsic_mat
    out["w"] = w
    out["h"] = h
    out["intrinsic_mat"] = intrinsic_mat.tolist()
    out["pointcloud_path"] = pointcloud_path
    out["frames"] = []
    get_extrinsic(out, extrinsic_path, image_path, name)

def get_intrinsic(intrinsic_path: str):
    with open(intrinsic_path) as f:
        angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            cx = w / 2
            cy = h / 2
            is_fisheye = False
            k1 = 0
            k2 = 0
            k3 = 0
            k4 = 0
            p1 = 0
            p2 = 0
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                is_fisheye = True
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL_FISHEYE":
                is_fisheye = True
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV_FISHEYE":
                is_fisheye = True
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                k3 = float(els[10])
                k4 = float(els[11])
            else:
                print("Unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    intrinsic_mat = np.eye(3, 3)
    intrinsic_mat[0, 0] = fl_x
    intrinsic_mat[1, 1] = fl_y
    intrinsic_mat[0, 2] = cx
    intrinsic_mat[1, 2] = cy
    return intrinsic_mat, w, h


def qvec2rotmat(qvec):
	""" 
	quaternion to RT matrix
	you can read more about quaternion in https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
	"""
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

def get_extrinsic(out: dict, extrinsic_path: str, IMAGE_FOLDER: str, OUT_PATH: str):
    with open(extrinsic_path) as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        

        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if  i % 2 == 1:
                elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                #name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(IMAGE_FOLDER)
                name = str(f"{'_'.join(elems[9:])}")
                # b = sharpness(name)
                # print(name, "sharpness=",b)
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                frame = {"file_path":name,"transform_matrix": c2w}
                out["frames"].append(frame)

        for f in out["frames"]:
            f["transform_matrix"] = f["transform_matrix"].tolist()

    sorted_frames = sorted(out["frames"], key=lambda x: x["file_path"])
    out["frames"] = sorted_frames
    
    print(f"writing {OUT_PATH}")
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)
    return out

def visualize_poses(poses, size=0.1):
    axes = trimesh.creation.axis(axis_length=.4) #, axis_radius=0.1)
    box = trimesh.primitives.Box(extents=(2.2, 2.2, 2.2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose, _ in poses:
        # a camera is visualized with 8 line segments.
        # pose = np.linalg.inv(pose)
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)
        
    return objects

def get_pointcloud(path):
    plydata = plyfile.PlyData.read(path)

    # Extract x, y, z coordinates
    x = plydata["vertex"]["x"]
    y = plydata["vertex"]["y"]
    z = plydata["vertex"]["z"]

    r = plydata["vertex"]["red"]
    g = plydata["vertex"]["green"]
    b = plydata["vertex"]["blue"]

    pointcloud = np.stack([x, y, z], axis=-1)
    colors = np.stack([r, g, b], axis=-1)

    return pointcloud, colors
