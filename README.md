# AI Storyteller – Interactive Fantasy Adventure Game

A full-stack application with a React & TypeScript frontend and FastAPI backend that dynamically generates branching fantasy stories using AI, complete with on-the-fly image creation.

---

## Tech Stack

**Frontend**  
- React 19 & TypeScript (Create React App)  
- Zustand for state management  
- Emotion (CSS-in-JS) for styling  
- React Router Dom for client-side routing  
- Testing Library & Jest for unit tests  

**Backend**  
- FastAPI + Uvicorn (ASGI server)  
- Pydantic for request/response models  
- CORS middleware  
- Python 3.8+  

**AI & Data**  
- **Story Generation**  
  1. Local LLM via llama-cpp-python (GGUF model)  
  2. Ollama (optional fallback)  
  3. HuggingFace pipeline (facebook/opt-350m)  
  4. Template-based fallback  
- **Retrieval Augmented Generation (RAG)**  
  - Embeddings: Sentence-Transformers (`all-MiniLM-L6-v2`)  
  - Vector store: ChromaDB  
  - Knowledge base: `server/knowledge_base/*.txt`  
- **Image Generation**  
  - Replicate API (Flux model: `black-forest-labs/flux-schnell`)  
  - Local placeholder on error  

**Tools & Infrastructure**  
- Node.js 14+ & npm  
- Python 3.8+ & pip  
- (Optional) CUDA for GPU acceleration  
- Git for source control  

---

## General Architecture & Flow

1. **User Interaction (Frontend)**  
   - Welcome → Character Creation → Gameplay screens  
   - On each turn, React posts to `/generate-story` with:  
     `character`, `story_start`, `location`, `previous_choice`  
   - When an image is needed, React posts to `/generate-image` with `story_text`.

2. **Story Generation (Backend)**  
   a. **RAG Retrieval**  
      - Load raw lore files from `server/knowledge_base`  
      - Compute embeddings with `all-MiniLM-L6-v2`  
      - Query ChromaDB for top-k relevant passages  
   b. **Prompt Assembly**  
      - Combine character profile + context + retrieved lore + last choice  
   c. **LLM Invocation** (in priority order)  
      1. **llama-cpp-python** (local GGUF model)  
      2. **Ollama** daemon (if installed)  
      3. **HuggingFace** OPT-350M  
      4. **Template fallback**  
   d. **Response Parsing**  
      - Extract narrative text, choice list, consequences, updated stats  

3. **Image Generation (Backend)**  
   - Compose an image prompt via `create_image_prompt()`  
   - If `REPLICATE_API_TOKEN` present:  
     • Instantiate `replicate.Client(api_token=…)`  
     • Run Flux model → receive image URL & local path  
   - Else: return a default `static/error.png`

4. **State & Caching**  
   - In-memory dicts track per-character story contexts & game state  
   - Simple TTL cache (`lru_cache`, custom TTL) prevents redundant RAG calls  

5. **Deployment**  
   - Frontend: `npm run build` → serve `build/` or via a static host  
   - Backend: `uvicorn app:app --host 0.0.0.0 --port 5000 --reload`  
   - Environment variables (see below)

---

## Prerequisites

- **Node.js** v14+ & **npm**  
- **Python** 3.8+ & **pip**  
- (Optional) **CUDA** for llama-cpp GPU acceleration  

---

## Getting Started

### Backend

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (Optional) download a local LLM model
python download_model.py

uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

### Frontend

```bash
cd ..
npm install
npm start
```

Visit `http://localhost:3000` for the React app (proxy to FastAPI at port 5000).

---

## Environment Variables

```bash
# Image generation
export REPLICATE_API_TOKEN="<your_replicate_token>"

# HuggingFace (if you plan to use HF-based fallbacks)
export HUGGINGFACE_API_TOKEN="<your_hf_token>"

# Local LLM settings
export LLAMA_MODEL_PATH="server/models/your_model.gguf"
export LLAMA_N_GPU_LAYERS=32  # number of layers on GPU
```
