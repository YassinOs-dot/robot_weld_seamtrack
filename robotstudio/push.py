# robotstudio/push.py
# Pushes RAPID .mod file to ABB RobotStudio via Robot Web Services (RWS)

import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RobotStudio Push Service")

RWS_HOST = os.environ.get("RWS_HOST", "192.168.125.1")
RWS_PORT = os.environ.get("RWS_PORT", "80")
RWS_USER = os.environ.get("RWS_USER", "Default User")
RWS_PASS = os.environ.get("RWS_PASS", "robotics")
BASE_URL = f"http://{RWS_HOST}:{RWS_PORT}"


class PushRequest(BaseModel):
    job_id    : str
    rapid_file: str


def rws(method, path, **kwargs):
    """Make an authenticated RWS request."""
    url  = f"{BASE_URL}{path}"
    auth = (RWS_USER, RWS_PASS)
    resp = requests.request(method, url, auth=auth, timeout=10, **kwargs)
    return resp


@app.get("/health")
def health():
    # test connection to robot controller
    try:
        r = rws("GET", "/rw/system")
        connected = r.status_code == 200
    except Exception:
        connected = False
    return {"status": "ok", "service": "robotstudio",
            "robot_connected": connected, "rws_host": RWS_HOST}


@app.post("/push")
def push(req: PushRequest):
    if not os.path.exists(req.rapid_file):
        raise HTTPException(400, f"RAPID file not found: {req.rapid_file}")

    with open(req.rapid_file, "r") as f:
        rapid_code = f.read()

    # ── 1. Request mastership ──────────────────────────────────
    r = rws("POST", "/rw/mastership",
            data={"action": "request"})
    if r.status_code not in (200, 204):
        raise HTTPException(502,
            f"Mastership request failed: {r.status_code} {r.text[:200]}")

    try:
        # ── 2. Upload module to controller ────────────────────
        r = rws("PUT",
                "/fileservice/HOME/WeldPath.mod",
                data=rapid_code.encode("utf-8"),
                headers={"Content-Type": "text/plain"})
        if r.status_code not in (200, 201, 204):
            raise HTTPException(502,
                f"File upload failed: {r.status_code} {r.text[:200]}")

        # ── 3. Load module into RAPID task ────────────────────
        r = rws("POST",
                "/rw/rapid/tasks/T_ROB1/program/modules",
                data={"modulepath": "HOME/WeldPath.mod",
                      "action": "load"})
        if r.status_code not in (200, 204):
            raise HTTPException(502,
                f"Module load failed: {r.status_code} {r.text[:200]}")

        # ── 4. Set program pointer to main ────────────────────
        r = rws("POST",
                "/rw/rapid/tasks/T_ROB1/pcp",
                data={"module": "WeldPath",
                      "routine": "main",
                      "action": "setpp"})
        if r.status_code not in (200, 204):
            raise HTTPException(502,
                f"Set PP failed: {r.status_code} {r.text[:200]}")

    finally:
        # ── 5. Release mastership always ──────────────────────
        rws("POST", "/rw/mastership",
            data={"action": "release"})

    return {
        "status"    : "ok",
        "job_id"    : req.job_id,
        "pushed_to" : RWS_HOST,
        "module"    : "WeldPath",
        "message"   : "Module loaded and PP set to main — ready to run"
    }
