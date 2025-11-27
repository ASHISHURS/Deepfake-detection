# main.py
import os
import math
import tempfile
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, SiglipForImageClassification

app = FastAPI(title="Video Deepfake Detector")

# === Model setup ===
MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model:", MODEL_NAME)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

ID2LABEL = {0: "fake", 1: "real"}

# === Folder setup for templates and static files ===
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# === Video frame sampling ===
def sample_frames_from_video(video_path: str, sample_fps: float = 1.0, max_frames: int = 64) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if fps > 0 else 0
    interval_frames = max(1, int(round(fps / sample_fps)))
    frames = []
    frame_idx = 0
    grabbed = True

    pbar = tqdm(total=min(max_frames, math.ceil(duration_sec * sample_fps)), desc="Sampling frames", leave=False)
    while grabbed and len(frames) < max_frames:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        if frame_idx % interval_frames == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            pbar.update(1)
        frame_idx += 1
    pbar.close()
    cap.release()
    return frames


# === Frame prediction ===
def predict_frames(frames: List[Image.Image], batch_size: int = 8):
    results = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            for p in probs:
                results.append({"prob_fake": float(p[0]), "prob_real": float(p[1])})
    return results


# === API endpoint to analyze video ===
@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...), sample_fps: float = 1.0, max_frames: int = 48):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        frames = sample_frames_from_video(tmp_path, sample_fps=sample_fps, max_frames=max_frames)
        if not frames:
            raise HTTPException(status_code=400, detail="No frames extracted from video")

        per_frame_scores = predict_frames(frames)

        avg_prob_fake = sum(f["prob_fake"] for f in per_frame_scores) / len(per_frame_scores)
        avg_prob_real = sum(f["prob_real"] for f in per_frame_scores) / len(per_frame_scores)

        if avg_prob_real >= avg_prob_fake:
            verdict = "real"
            confidence = float(avg_prob_real)
        else:
            verdict = "fake"
            confidence = float(avg_prob_fake)

        response = {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "avg_prob_real": round(avg_prob_real, 4),
            "avg_prob_fake": round(avg_prob_fake, 4),
            "sampled_frames": len(per_frame_scores),
            "per_frame": [
                {"idx": i, "prob_fake": round(s["prob_fake"], 4), "prob_real": round(s["prob_real"], 4)}
                for i, s in enumerate(per_frame_scores[:20])
            ]
        }
        return JSONResponse(response)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# === Serve HTML page ===
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# === Run app ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8050, reload=True)
