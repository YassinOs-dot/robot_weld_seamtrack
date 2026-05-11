# api/main.py
# Central orchestrator — receives images, runs full pipeline

import os
import shutil
import asyncio
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import List
import uuid

app        = FastAPI(title="RoboWeldAR API", version="1.0.0")
SHARED_DIR = os.environ.get("SHARED_DIR", "/shared")
RECON_URL  = os.environ.get("RECON_URL", "http://reconstruction:8001")
SEAM_URL   = os.environ.get("SEAM_URL",  "http://seam_detection:8002")
TRAJ_URL   = os.environ.get("TRAJ_URL",  "http://trajectory:8003")
RAPID_URL  = os.environ.get("RAPID_URL", "http://rapid_export:8004")
ROBOT_URL  = os.environ.get("ROBOT_URL", "http://robotstudio:8005")

# ── job store (in-memory — replace with Redis for production) ──
jobs = {}


def job_dir(job_id):
    return os.path.join(SHARED_DIR, "jobs", job_id)


@app.get("/health")
def health():
    return {"status": "ok", "service": "api"}


@app.post("/pipeline/start")
async def start_pipeline(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    send_to_robot: bool = True
):
    """
    Upload images → run full pipeline → optionally push to RobotStudio.
    Returns job_id for status polling.
    """
    job_id  = str(uuid.uuid4())[:8]
    jdir    = job_dir(job_id)
    img_dir = os.path.join(jdir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # save uploaded images
    for img in images:
        dest = os.path.join(img_dir, img.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(img.file, f)

    jobs[job_id] = {"status": "queued", "step": None, "error": None}
    background_tasks.add_task(
        run_pipeline, job_id, jdir, send_to_robot
    )
    return {"job_id": job_id, "images": len(images), "status": "queued"}


@app.get("/pipeline/{job_id}/status")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/pipeline/{job_id}/download/{filename}")
def download_file(job_id: str, filename: str):
    path = os.path.join(job_dir(job_id), filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"{filename} not found for job {job_id}")
    return FileResponse(path, filename=filename)


async def run_pipeline(job_id: str, jdir: str, send_to_robot: bool):
    """Orchestrates all microservices in sequence."""
    jobs[job_id]["status"] = "running"

    async with httpx.AsyncClient(timeout=300.0) as client:

        # ── Step 1: 3D reconstruction ──────────────────────────
        jobs[job_id]["step"] = "reconstruction"
        try:
            r = await client.post(f"{RECON_URL}/reconstruct", json={
                "job_id"  : job_id,
                "img_dir" : os.path.join(jdir, "images"),
                "out_dir" : jdir
            })
            r.raise_for_status()
        except Exception as e:
            jobs[job_id].update({"status": "failed", "error": f"reconstruction: {e}"})
            return

        # ── Step 2: Seam detection ─────────────────────────────
        jobs[job_id]["step"] = "seam_detection"
        try:
            r = await client.post(f"{SEAM_URL}/detect", json={
                "job_id" : job_id,
                "obj_file": os.path.join(jdir, "mesh.obj"),
                "out_dir" : jdir
            })
            r.raise_for_status()
        except Exception as e:
            jobs[job_id].update({"status": "failed", "error": f"seam_detection: {e}"})
            return

        # ── Step 3: Trajectory planning ────────────────────────
        jobs[job_id]["step"] = "trajectory"
        try:
            r = await client.post(f"{TRAJ_URL}/plan", json={
                "job_id"          : job_id,
                "centerline_file" : os.path.join(jdir, "seam_centerline.json"),
                "out_dir"         : jdir
            })
            r.raise_for_status()
        except Exception as e:
            jobs[job_id].update({"status": "failed", "error": f"trajectory: {e}"})
            return

        # ── Step 4: RAPID export ───────────────────────────────
        jobs[job_id]["step"] = "rapid_export"
        try:
            r = await client.post(f"{RAPID_URL}/export", json={
                "job_id"         : job_id,
                "waypoints_file" : os.path.join(jdir, "weld_waypoints.json"),
                "out_dir"        : jdir
            })
            r.raise_for_status()
        except Exception as e:
            jobs[job_id].update({"status": "failed", "error": f"rapid_export: {e}"})
            return

        # ── Step 5: Push to RobotStudio (optional) ─────────────
        if send_to_robot:
            jobs[job_id]["step"] = "robotstudio"
            try:
                r = await client.post(f"{ROBOT_URL}/push", json={
                    "job_id"     : job_id,
                    "rapid_file" : os.path.join(jdir, "weld_path.mod"),
                })
                r.raise_for_status()
            except Exception as e:
                jobs[job_id].update({
                    "status": "failed",
                    "error" : f"robotstudio: {e}"
                })
                return

    jobs[job_id].update({
        "status"     : "done",
        "step"       : "complete",
        "rapid_file" : os.path.join(jdir, "weld_path.mod"),
        "waypoints"  : os.path.join(jdir, "weld_waypoints.json")
    })
