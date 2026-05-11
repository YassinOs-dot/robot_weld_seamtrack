# trajectory/plan.py

import os, json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import splprep, splev
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Trajectory Service")

N_SMOOTH         = int(os.environ.get("N_SMOOTH",         "100"))
WELD_SPEED       = int(os.environ.get("WELD_SPEED",       "8"))
APPROACH_SPEED   = int(os.environ.get("APPROACH_SPEED",   "50"))
RETRACT_SPEED    = int(os.environ.get("RETRACT_SPEED",    "50"))
APPROACH_DIST    = float(os.environ.get("APPROACH_DIST",  "15"))
RETRACT_DIST     = float(os.environ.get("RETRACT_DIST",   "15"))
WORK_ANGLE       = float(os.environ.get("WORK_ANGLE",     "0"))
TRAVEL_ANGLE     = float(os.environ.get("TRAVEL_ANGLE",   "5"))


class PlanRequest(BaseModel):
    job_id          : str
    centerline_file : str
    out_dir         : str


def to_python(obj):
    if isinstance(obj, dict):          return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):          return [to_python(v) for v in obj]
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.bool_):      return bool(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    return obj


def chord_param(pts):
    diffs  = np.diff(pts, axis=0)
    dists  = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(dists)])
    total  = cumlen[-1]
    return cumlen / total if total > 1e-9 else np.linspace(0, 1, len(pts))


def smooth_path(pts, n_out):
    if len(pts) < 4:
        t   = np.linspace(0, 1, n_out)
        t0  = chord_param(pts)
        out = np.column_stack([np.interp(t, t0, pts[:, d]) for d in range(3)])
        tan = np.gradient(out, axis=0)
        tan = tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9)
        return out, tan
    u   = chord_param(pts)
    tck, _ = splprep([pts[:,0], pts[:,1], pts[:,2]], u=u,
                     s=len(pts)*0.2, k=3)
    u_o    = np.linspace(0, 1, n_out)
    xs,ys,zs   = splev(u_o, tck)
    dxs,dys,dzs = splev(u_o, tck, der=1)
    out  = np.column_stack([xs, ys, zs])
    tan  = np.column_stack([dxs, dys, dzs])
    tan  = tan / (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9)
    return out, tan


def build_frame(tangent, surf_normal, work_deg, travel_deg):
    torch_z = -surf_normal / (np.linalg.norm(surf_normal) + 1e-9)
    torch_x = tangent - np.dot(tangent, torch_z) * torch_z
    n = np.linalg.norm(torch_x)
    torch_x = torch_x / n if n > 1e-9 else np.array([1., 0., 0.])
    torch_y = np.cross(torch_z, torch_x)
    torch_y = torch_y / (np.linalg.norm(torch_y) + 1e-9)
    torch_x = np.cross(torch_y, torch_z)
    torch_x = torch_x / (np.linalg.norm(torch_x) + 1e-9)
    R = np.column_stack([torch_x, torch_y, torch_z])
    Rw = Rotation.from_rotvec(np.radians(work_deg) * torch_x).as_matrix()
    Rt = Rotation.from_rotvec(np.radians(travel_deg) * torch_y).as_matrix()
    return Rt @ Rw @ R


@app.get("/health")
def health():
    return {"status": "ok", "service": "trajectory"}


@app.post("/plan")
def plan(req: PlanRequest):
    if not os.path.exists(req.centerline_file):
        raise HTTPException(400, f"File not found: {req.centerline_file}")

    with open(req.centerline_file) as f:
        data = json.load(f)

    centerlines  = data.get("centerlines") or [data.get("centerline", [])]
    groove_norms = data.get("groove_normals", [])
    surf_normal  = (np.array(groove_norms[0])
                    if groove_norms else np.array([0., 0., 1.]))

    all_paths = []
    for si, cl_raw in enumerate(centerlines):
        pts = np.array(cl_raw)
        if len(pts) == 0:
            continue

        smooth_pts, tangents = smooth_path(pts, N_SMOOTH)

        rots = [Rotation.from_matrix(
            build_frame(t, surf_normal, WORK_ANGLE, TRAVEL_ANGLE)
        ) for t in tangents]

        key_t  = np.linspace(0, 1, len(rots))
        slerp  = Slerp(key_t, Rotation.concatenate(rots))
        s_rots = slerp(key_t)

        weld_wps = [{
            "index"     : int(i),
            "type"      : "weld",
            "seam_index": si,
            "position"  : pt.tolist(),
            "quaternion": rot.as_quat().tolist(),
            "euler_deg" : rot.as_euler('xyz', degrees=True).tolist(),
            "speed"     : WELD_SPEED
        } for i, (pt, rot) in enumerate(zip(smooth_pts, s_rots))]

        app_wp = {
            "index": -1, "type": "approach", "seam_index": si,
            "position": (smooth_pts[0] + [0,0,APPROACH_DIST]).tolist(),
            "quaternion": weld_wps[0]["quaternion"],
            "euler_deg": weld_wps[0]["euler_deg"],
            "speed": APPROACH_SPEED
        }
        ret_wp = {
            "index": len(weld_wps), "type": "retract", "seam_index": si,
            "position": (smooth_pts[-1] + [0,0,RETRACT_DIST]).tolist(),
            "quaternion": weld_wps[-1]["quaternion"],
            "euler_deg": weld_wps[-1]["euler_deg"],
            "speed": RETRACT_SPEED
        }
        all_paths.append([app_wp] + weld_wps + [ret_wp])

    os.makedirs(req.out_dir, exist_ok=True)
    out_file = os.path.join(req.out_dir, "weld_waypoints.json")
    output   = to_python({
        "job_id"     : req.job_id,
        "num_seams"  : len(all_paths),
        "seam_paths" : all_paths,
        "all_waypoints": [wp for path in all_paths for wp in path]
    })
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    return {"status": "ok", "seams": len(all_paths), "output": out_file}
