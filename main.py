from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import List

from clip_processor import clip_processor
from magic_processor import magic_processor
from removebg_processor import removebg_processor

app = FastAPI(title="MagicMirror Backend API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "MagicMirror Backend Service Ready"}

# ================= CLIP / Semantic Classification =================

@app.post("/api/clip/classify")
async def clip_classify(
    image: UploadFile = File(...),
    candidates: str = Form(..., description="Comma separated list of categories, e.g., 'cat,dog,car'")
):
    """
    Classify an image into one of the provided candidate categories.
    """
    try:
        image_bytes = await image.read()
        candidate_list = [c.strip() for c in candidates.split(",") if c.strip()]
        
        if not candidate_list:
            raise HTTPException(status_code=400, detail="Candidate list cannot be empty")
            
        # Construct prompts
        prompts = [f"一张{c}的照片" for c in candidate_list]
        
        probs, logits = clip_processor.predict(image_bytes, prompts)
        
        # Format results
        results = []
        for i, label in enumerate(candidate_list):
            results.append({"label": label, "score": probs[i]})
            
        # Sort by score desc
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "best_match": results[0]["label"],
            "best_score": results[0]["score"],
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clip/match")
async def clip_match(
    image: UploadFile = File(...),
    target: str = Form(..., description="Target object name, e.g., 'shoes'"),
    threshold: float = Form(0.6)
):
    """
    Check if the image contains the target object.
    """
    try:
        image_bytes = await image.read()
        
        positive_text = f"一张{target}的照片"
        negative_text = "一张其他物品的照片"
        texts = [positive_text, negative_text]
        
        probs, _ = clip_processor.predict(image_bytes, texts)
        
        pos_score = probs[0]
        is_match = pos_score > threshold
        
        return {
            "target": target,
            "is_match": is_match,
            "confidence": pos_score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================= Magic / Inpainting =================

@app.post("/api/magic/inpaint")
async def magic_inpaint(
    image: UploadFile = File(...)
):
    # TODO: Handle mask upload or generation
    image_bytes = await image.read()
    return magic_processor.process(image_bytes)

# ================= RemoveBG =================

@app.post("/api/removebg")
async def remove_background(
    image: UploadFile = File(...)
):
    image_bytes = await image.read()
    return removebg_processor.process(image_bytes)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
