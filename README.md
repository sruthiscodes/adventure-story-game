# AI Storyteller – Interactive Fantasy Adventure Game

A multimodal full-stack interactive fantasy adventure game that generates dynamic, branching narratives using AI. Built with React & TypeScript on the frontend and FastAPI on the backend, this project allows users to embark on personalized storytelling journeys, where their choices shape the adventure. Additionally, the game includes AI-driven image generation, creating visuals to match key moments in the story, providing both text and visual elements for an enriched, immersive experience.


## Tech stack
- React (TypeScript)
- Python (FastAPI)
- Local LLMs via `llama-cpp-python` (GGUF), with fallbacks to Ollama, HuggingFace (`facebook/opt-350m`), and local templates  
- Image generation via Replicate (Flux: `black-forest-labs/flux-schnell`)
- RAG (Retrieval-Augmented Generation) using Sentence-Transformers (`all-MiniLM-L6-v2`), and ChromaDB


## Prerequisites

- **Node.js** v14+ & **npm**  
- **Python** 3.8+ & **pip**  
- (Optional) **CUDA** for llama-cpp GPU acceleration  


## Getting started

### Set environment variables
First, set your environment variables for Replicate and HuggingFace API tokens:

```bash
#Set Replicate API token for image generation
export REPLICATE_API_TOKEN="<your_replicate_token>"
#Set HuggingFace API token for text generation and embedding for RAG system
export HUGGINGFACE_API_TOKEN="<your_hf_token>"
```

### Backend 
1. Open a terminal window and navigate to the project directory.

2. Set up a Python virtual environment:
```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Start the FastAPI backend:
```bash
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

This runs the backend API on `http://localhost:5000`

### Frontend
1. In a new terminal window, navigate to the frontend directory and install dependencies:

```bash
cd ..
npm install
```

2. Start the React frontend:
```bash
npm start
```

The web app is hosted on `http://localhost:3000`

