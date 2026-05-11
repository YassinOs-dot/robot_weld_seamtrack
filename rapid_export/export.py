# rapid_export/export.py

import os, json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RAPID Export Service")

TOOL_NAME  = os.environ.get("TOOL_NAME",  "tWeldGun")
WORK_OBJ   = os.environ.get("WORK_OBJ",   "wobjWeldTable")
WELD_SPEED = int(os.environ.get("WELD_SPEED", "8"))


class ExportRequest(BaseModel):
    job_id        : str
    waypoints_file: str
    out_dir       : str


def quat_to_abb(q):
    x, y, z, w = q
    return [w, x, y, z]


def robtarget(name, pos, q_abb):
    px,py,pz    = pos
    q1,q2,q3,q4 = q_abb
    return (f"    CONST robtarget {name}:="
            f"[[{px:.4f},{py:.4f},{pz:.4f}],"
            f"[{q1:.6f},{q2:.6f},{q3:.6f},{q4:.6f}],"
            f"[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];")


@app.get("/health")
def health():
    return {"status": "ok", "service": "rapid_export"}


@app.post("/export")
def export(req: ExportRequest):
    if not os.path.exists(req.waypoints_file):
        raise HTTPException(400, f"File not found: {req.waypoints_file}")

    with open(req.waypoints_file) as f:
        data = json.load(f)

    seam_paths = data.get("seam_paths", [])
    if not seam_paths:
        all_wps = data.get("all_waypoints", [])
        seam_paths = [all_wps] if all_wps else []

    all_decls, all_procs = [], []

    for si, path in enumerate(seam_paths):
        decls, moves = [], []
        for wp in path:
            pos   = wp["position"]
            q_abb = quat_to_abb(wp["quaternion"])
            spd   = wp.get("speed", WELD_SPEED)

            if wp["type"] == "approach":
                name = f"pS{si+1}App"
            elif wp["type"] == "retract":
                name = f"pS{si+1}Ret"
            else:
                name = f"pS{si+1}_{wp['index']:04d}"

            decls.append(robtarget(name, pos, q_abb))

            if wp["type"] in ("approach", "retract"):
                moves.append(
                    f"        MoveL {name},v{spd},z5,"
                    f"{TOOL_NAME}\\WObj:={WORK_OBJ};"
                )
            else:
                moves.append(
                    f"        ArcL {name},v{spd},"
                    f"seam{si+1},weld{si+1},z1,"
                    f"{TOOL_NAME}\\WObj:={WORK_OBJ};"
                )

        # ArcLStart / ArcLEnd
        fw = next(i for i,w in enumerate(path) if w["type"]=="weld")
        lw = len(path)-1-next(
            i for i,w in enumerate(reversed(path)) if w["type"]=="weld"
        )
        n_fw = f"pS{si+1}_{path[fw]['index']:04d}"
        n_lw = f"pS{si+1}_{path[lw]['index']:04d}"
        moves[fw] = (f"        ArcLStart {n_fw},v{WELD_SPEED},"
                     f"seam{si+1},weld{si+1},z1,"
                     f"{TOOL_NAME}\\WObj:={WORK_OBJ};")
        moves[lw] = (f"        ArcLEnd {n_lw},v{WELD_SPEED},"
                     f"seam{si+1},weld{si+1},z1,"
                     f"{TOOL_NAME}\\WObj:={WORK_OBJ};")

        all_decls.extend(decls)
        all_procs.append(
            f"\n    PROC WeldSeam{si+1}()\n"
            + "\n".join(moves)
            + "\n    ENDPROC"
        )

    master = "\n".join(
        f"        WeldSeam{i+1}();"
        for i in range(len(seam_paths))
    )

    seam_data = "\n".join(
        f"    PERS seamdata seam{i+1}:=[0,0.2,0,0,0,0];\n"
        f"    PERS welddata weld{i+1}:=[0.2,1,0,0,0,0,0,0,0,0];"
        for i in range(len(seam_paths))
    )

    rapid = f"""MODULE WeldPath
! Auto-generated — RoboWeldAR Docker pipeline
! Standard: ISO 5817-D  Process: MAG 135
! Tool: {TOOL_NAME}  WObj: {WORK_OBJ}

{seam_data}

{chr(10).join(all_decls)}

    PROC main()
{master}
    ENDPROC
{"".join(all_procs)}

ENDMODULE
"""
    os.makedirs(req.out_dir, exist_ok=True)
    out_file = os.path.join(req.out_dir, "weld_path.mod")
    with open(out_file, "w") as f:
        f.write(rapid)

    return {"status": "ok", "rapid_file": out_file,
            "seams": len(seam_paths),
            "total_robtargets": len(all_decls)}
