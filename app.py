from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
 
app = FastAPI(title="LLM Service", description="FastAPI wrapper around Ollama LLM server")
 
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:0.5b"
 
 
class PromptRequest(BaseModel):
    """Request model for LLM inference."""
    prompt: str
    model: str = DEFAULT_MODEL
 
 
class PromptResponse(BaseModel):
    """Response model for LLM inference."""
    prompt: str
    response: str
    model: str
 
 
@app.get("/health")
def health_check() -> dict:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        return {"status": "ok", "ollama": "running"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {e}")
 
 
@app.post("/generate", response_model=PromptResponse)
def generate(request: PromptRequest) -> PromptResponse:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Ollama request failed: {e}")
 
    data = r.json()
    return PromptResponse(
        prompt=request.prompt,
        response=data["response"],
        model=request.model,
    )
 