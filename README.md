# AI Storyteller – Interactive Fantasy Adventure Game

A multimodal full-stack interactive fantasy adventure game that generates dynamic, branching narratives using AI. Built with React & TypeScript on the frontend and FastAPI on the backend, this project allows users to embark on personalized storytelling journeys, where their choices shape the adventure. Additionally, the game includes AI-driven image generation, creating visuals to match key moments in the story, providing both text and visual elements for an enriched, immersive experience.

---

## Tech Stack
- React 19 (TypeScript)
- Python 3.8+ (FastAPI)
- Local LLMs via `llama-cpp-python` (GGUF), with fallbacks to Ollama, HuggingFace (`facebook/opt-350m`), and local templates  
- Image generation via Replicate (Flux: `black-forest-labs/flux-schnell`)
- RAG using Sentence-Transformers (`all-MiniLM-L6-v2`), and ChromaDB

---



## Prerequisites

- **Node.js** v14+ & **npm**  
- **Python** 3.8+ & **pip**  
- (Optional) **CUDA** for llama-cpp GPU acceleration  

---

## Getting Started

## Set environment variables

```bash
#Set Replicate API token for image generation
export REPLICATE_API_TOKEN="<your_replicate_token>"
#Set HuggingFace API token for text generation and embedding for RAG system
export HUGGINGFACE_API_TOKEN="<your_hf_token>"
```

### Backend

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

### Frontend

```bash
cd ..
npm install
npm start
```

The web app is hosted on `http://localhost:3000`
---


