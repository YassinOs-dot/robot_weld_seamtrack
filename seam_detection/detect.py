# seam_detection/detect.py
# Gap-based butt weld seam detection from mesh/point cloud
# Implements: symmetry edge detector (Merium88) + gap void detector

import os
import json
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
os.environ["OPEN3D_DISABLE_WEB_VISUALIZER"] = "1"
app = FastAPI(title="Seam Detection Service")

# ── CONFIG (tunable via env vars) ─────────────────────────────
VOXEL             = float(os.environ.get("VOXEL",            "0.2"))
EDGE_K            = int(os.environ.get("EDGE_K",             "20"))
EDGE_RATIO        = float(os.environ.get("EDGE_RATIO",       "2.0"))
GAP_VOID_THRESH   = float(os.environ.get("GAP_VOID_THRESH",  "0.25"))
SEAM_SEARCH_R     = float(os.environ.get("SEAM_SEARCH_R",    "3.0"))
THINNING_K        = int(os.environ.get("THINNING_K",         "8"))
N_SAMPLE          = int(os.environ.get("N_SAMPLE",           "20000"))


class DetectRequest(BaseModel):
    job_id  : str
    obj_file: str
    out_dir : str


def to_python(obj):
    if isinstance(obj, dict):          return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):          return [to_python(v) for v in obj]
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.bool_):      return bool(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    return obj


def detect_edges(points, k=20, ratio=2.0):
    tree       = KDTree(points)
    dists, idx = tree.query(points, k=k)
    nn_dist    = dists[:, 1]
    centroids  = points[idx].mean(axis=1)
    asym       = np.linalg.norm(points - centroids, axis=1)
    return np.where(asym > nn_dist * ratio)[0]


def detect_gaps(points, void_thresh=0.25):
    tree     = KDTree(points)
    dists, _ = tree.query(points, k=8)
    max_gap  = dists[:, 1:].max(axis=1)
    return np.where(max_gap > void_thresh)[0]


def thin_order(pts, k=8):
    if len(pts) < k:
        return pts
    tree     = KDTree(pts)
    _, idx   = tree.query(pts, k=k)
    thinned  = pts[idx].mean(axis=1)
    _, uniq  = np.unique(np.round(thinned, 2), axis=0, return_index=True)
    thinned  = thinned[uniq]
    if len(thinned) < 2:
        return thinned
    tree2    = KDTree(thinned)
    visited  = np.zeros(len(thinned), dtype=bool)
    start    = int(np.argmin(thinned[:, 0]))
    ordered  = [start]
    visited[start] = True
    for _ in range(len(thinned) - 1):
        cur      = ordered[-1]
        _, cands = tree2.query(thinned[cur], k=min(15, len(thinned)))
        for j in cands:
            if not visited[j]:
                ordered.append(j)
                visited[j] = True
                break
    return thinned[ordered]


def local_tangents(pts, k=8):
    tree   = KDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)))
    tans   = []
    for i in range(len(pts)):
        nn = pts[idx[i]]
        c  = nn - nn.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(c, full_matrices=False)
            t = Vt[0]
        except Exception:
            t = np.array([1., 0., 0.])
        if tans and np.dot(t, tans[-1]) < 0:
            t = -t
        tans.append(t)
    return np.array(tans)


def torch_frame(tangent, surf_normal):
    torch_z = -surf_normal / (np.linalg.norm(surf_normal) + 1e-9)
    torch_x = tangent - np.dot(tangent, torch_z) * torch_z
    n = np.linalg.norm(torch_x)
    torch_x = torch_x / n if n > 1e-9 else np.array([1., 0., 0.])
    torch_y = np.cross(torch_z, torch_x)
    torch_y = torch_y / (np.linalg.norm(torch_y) + 1e-9)
    torch_x = np.cross(torch_y, torch_z)
    torch_x = torch_x / (np.linalg.norm(torch_x) + 1e-9)
    return np.column_stack([torch_x, torch_y, torch_z])


@app.get("/health")
def health():
    return {"status": "ok", "service": "seam_detection"}


@app.post("/detect")
def detect(req: DetectRequest):
    if not os.path.exists(req.obj_file):
        raise HTTPException(400, f"OBJ not found: {req.obj_file}")

    os.makedirs(req.out_dir, exist_ok=True)

    # load mesh → point cloud
    mesh = o3d.io.read_triangle_mesh(req.obj_file)
    mesh.compute_vertex_normals()
    n_pts = min(N_SAMPLE, len(mesh.triangles) * 5)
    pcd   = mesh.sample_points_uniformly(number_of_points=n_pts)
    pcd, _  = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd   = pcd.voxel_down_sample(voxel_size=VOXEL)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=VOXEL * 3, max_nn=20
        )
    )
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 500.])
    )

    points  = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # detect edge + gap points
    e_idx = detect_edges(points, k=EDGE_K, ratio=EDGE_RATIO)
    g_idx = detect_gaps(points, void_thresh=GAP_VOID_THRESH)

    if len(e_idx) > 0 and len(g_idx) > 0:
        combined = np.vstack([points[e_idx], points[g_idx]])
    elif len(e_idx) > 0:
        combined = points[e_idx]
    elif len(g_idx) > 0:
        combined = points[g_idx]
    else:
        raise HTTPException(422, "No seam candidates detected — "
                                 "try adjusting EDGE_RATIO or GAP_VOID_THRESH")

    centerline = thin_order(combined, k=THINNING_K)

    if len(centerline) < 2:
        raise HTTPException(422, "Centerline too short — "
                                 "check mesh quality or detection parameters")

    tangents    = local_tangents(centerline, k=8)
    surf_normal = normals.mean(axis=0)
    surf_normal = surf_normal / (np.linalg.norm(surf_normal) + 1e-9)

    waypoints = []
    for i, (pt, tan) in enumerate(zip(centerline, tangents)):
        R    = torch_frame(tan, surf_normal)
        rot  = Rotation.from_matrix(R)
        waypoints.append({
            "index"     : int(i),
            "position"  : pt.tolist(),
            "quaternion": rot.as_quat().tolist(),
            "euler_deg" : rot.as_euler('xyz', degrees=True).tolist(),
            "tangent"   : tan.tolist()
        })

    out = to_python({
        "job_id"       : req.job_id,
        "num_waypoints": len(waypoints),
        "centerline"   : centerline.tolist(),
        "groove_normals": [surf_normal.tolist()],
        "waypoints"    : waypoints,
        "seam_results" : [{
            "seam_index": 0,
            "n_waypoints": len(waypoints),
            "source": "detected"
        }]
    })

    out_file = os.path.join(req.out_dir, "seam_centerline.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)

    return {"status": "ok", "seams": 1,
            "waypoints": len(waypoints), "output": out_file}
