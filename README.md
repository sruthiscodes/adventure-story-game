# AI Storyteller – Interactive Fantasy Adventure Game

A full-stack application with a React & TypeScript frontend and FastAPI backend that dynamically generates branching fantasy stories using AI, complete with on-the-fly image creation.

## Prerequisites

Before you begin, ensure you have:

•  **Node.js** v14+  
•  **npm** (comes with Node.js)  
•  **Python** 3.8+  
•  **pip** (comes with Python)  
•  (Optional) **CUDA** setup if you plan to run local Llama models on GPU  

## Project Setup

### 1. Backend (FastAPI)

```bash
# 1. Enter the server folder
cd server

# 2. (Recommended) Create & activate a virtual env
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download a local Llama GGUF model
python download_model.py

# 5. Start the backend server
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

The backend will be available at `http://localhost:5000`.

### 2. Frontend (React)

In a separate terminal:

```bash
# 1. From project root
cd .

# 2. Install Node dependencies
npm install

# 3. Start the development server
npm start
```

Your React app will launch at `http://localhost:3000` and proxy API requests to the FastAPI backend.

## Usage

1. **New Game**  
   - Click **“New Adventure”** on the welcome screen.  
   - Fill in your character’s name and background.

2. **Gameplay**  
   - Make choices as the narrative unfolds.  
   - Watch your character stats update and the story branch in real time.

3. **Image Generation**  
   - Each story segment may generate an AI-produced image.  
   - Images are stored in the server’s `generated_images/` folder.

4. **Emergency Controls**  
   - “Reset Game” clears all saved state.  
   - “Fix Game Flow” jumps you back to character creation.

## Environment Variables

- **REPLICATE_API_TOKEN**  
  For image generation via Replicate.
- **LLAMA_MODEL_PATH**  
  Path to your local GGUF model if using llama-cpp.
- **LLAMA_N_GPU_LAYERS**  
  Number of layers to offload to GPU (default: 0 for CPU).

Example (macOS/Linux):

```bash
export REPLICATE_API_TOKEN="your_token_here"
export LLAMA_MODEL_PATH="server/models/your_model.gguf"
export LLAMA_N_GPU_LAYERS=32
```
