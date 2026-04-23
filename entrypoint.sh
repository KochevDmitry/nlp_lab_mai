#!/bin/bash

ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama to start"
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama is ready."

echo "Pulling qwen2.5:0.5b..."
ollama pull qwen2.5:0.5b
echo "Model ready."

echo "Starting FastAPI"
uvicorn app:app --host 0.0.0.0 --port 8000