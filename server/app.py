import os
import json
import io
import base64
import requests
import random
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from functools import lru_cache
import datetime
import aiohttp
import re
import logging
from pathlib import Path
import replicate
import traceback
from uuid import uuid4

# Initialize FastAPI app and configure CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define static directories for serving files
OUTPUT_DIR = "generated_images"
STATIC_DIR = "static"
# Ensure directories exist before mounting
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static directories for serving files
app.mount("/generated_images", StaticFiles(directory=OUTPUT_DIR), name="generated_images")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize llama.cpp model
LLAMA_CPP_AVAILABLE = False
LLAMA_MODEL_INITIALIZED = False
try:
    from llama_cpp import Llama
    LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/llama_model.gguf")
    LLAMA_MODEL = Llama(model_path=LLAMA_MODEL_PATH, n_ctx=2048)
    LLAMA_CPP_AVAILABLE = True
    LLAMA_MODEL_INITIALIZED = True
    print(f"Initialized llama.cpp model from {LLAMA_MODEL_PATH}")
except Exception as e:
    print(f"Failed to initialize llama.cpp: {e}")

# Define Pydantic models
class CharacterModel(BaseModel):
    name: str
    background: Optional[str] = None
    stats: Optional[Dict[str, int]] = {
        "strength": 10,
        "wisdom": 10,
        "agility": 10,
        "charisma": 10,
        "health": 100,
        "magic": 10
    }

class ConsequenceModel(BaseModel):
    stat: Optional[str] = "none"  # The stat to affect: strength, wisdom, agility, charisma, health, magic
    change: Optional[int] = 0     # The amount to change (positive or negative)
    description: Optional[str] = ""  # Description of the consequence

class ChoiceModel(BaseModel):
    text: str
    consequence: Optional[ConsequenceModel] = None

class StoryRequest(BaseModel):
    character: Optional[CharacterModel] = None
    character_name: Optional[str] = None
    previous_choice: Optional[str] = None
    story_start: Optional[bool] = False
    location: Optional[str] = None

class StoryResponse(BaseModel):
    content: str
    choices: List[ChoiceModel]
    character: Optional[CharacterModel] = None
    image_prompt: str
    story: str
    options: List[str]

class ImageRequest(BaseModel):
    story_text: str
    character_name: Optional[str] = None

class ImageResponse(BaseModel):
    image_url: str
    prompt: str
    success: bool
    generation_method: str
    image_path: str

# Global storage for story contexts and game states
story_contexts = {}
GAME_STATES = {}

# Import RAG components
import chromadb
from sentence_transformers import SentenceTransformer

# Import HuggingFace components
import torch
from transformers import pipeline as hf_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Update story generation settings
MAX_STORY_WORDS = 75  # Reduced from 300 to 75 words for shorter story points
MAX_TOKENS = 150  # Reduced token count for more concise stories

# Initialize HuggingFace Transformers model
use_cuda = torch.cuda.is_available()
device_hf = 0 if use_cuda else -1
story_generator = hf_pipeline(
    "text-generation",
    model="facebook/opt-350m",  # Use OPT model instead of gpt2 for better quality
    framework="pt",
    device=device_hf,
    truncation=True,
    max_new_tokens=MAX_TOKENS,
    pad_token_id=50256
)

# RAG Setup
KNOWLEDGE_DIR = "knowledge_base"
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# Initialize RAG components
rag_enabled = False
embedding_model = None
vector_db = None

try:
    # Initialize embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize vector database
    vector_db = chromadb.Client()
    fantasy_collection = vector_db.create_collection(name="fantasy_lore")
    
    # Create sample fantasy knowledge if it doesn't exist
    sample_lore_files = [
        ("races.txt", "Fantasy Races and Species", 
         "Elves: Known for their grace, pointed ears, and long lifespans. They excel in magic and archery, and typically live in forest settlements.\n"
         "Dwarves: Stout and hardy folk who live in mountain halls. Expert miners, crafters, and warriors. Known for their beards and love of gold.\n"
         "Humans: Adaptable and ambitious, found in all walks of life. Their kingdoms are vast and their cultures diverse.\n"
         "Orcs: Strong tribal warriors with tusked jaws. Often considered brutal and primitive, but have complex clan structures.\n"
         "Goblins: Small, cunning creatures that live in caves and underground lairs. Known for traps and ambushes.\n"
         "Dragons: Ancient, intelligent reptilian creatures with wings and breath weapons. Often hoard treasures and possess powerful magic."),
        
        ("magic.txt", "Magic Systems", 
         "Elemental Magic: Control over fire, water, earth, air, and sometimes light and shadow. Most common and accessible form of magic.\n"
         "Divine Magic: Granted by gods or spiritual entities. Requires faith and devotion. Often used for healing and protection.\n"
         "Arcane Magic: Academic magic requiring study and understanding of complex formulas. Uses components and often requires spellbooks.\n"
         "Blood Magic: Powerful forbidden magic using life force. Often considered taboo due to its dark nature.\n"
         "Wild Magic: Unpredictable, nature-based magic. Sometimes manifests spontaneously in certain regions or bloodlines."),
        
        ("locations.txt", "Fantasy Locations", 
         "Enchanted Forests: Ancient woodlands filled with magical creatures, fairies, and sometimes darker entities. Trees might be sentient.\n"
         "Forgotten Ruins: Abandoned cities or temples containing ancient knowledge, traps, and treasures.\n"
         "Mountain Citadels: Fortresses built into mountainsides, often home to dwarves or secluded wizards.\n"
         "Desert Oases: Magical pools that sustain life in barren lands, sometimes gateways to other realms.\n"
         "Underwater Kingdoms: Vast civilizations beneath the waves, home to merfolk and other aquatic species.\n"
         "Shadow Realms: Parallel dimensions of darkness where normal rules don't apply. Home to demons and nightmares.")
    ]
    
    # Create knowledge base files if they don't exist
    lore_documents = []
    lore_metadata = []
    lore_ids = []
    
    for i, (filename, title, content) in enumerate(sample_lore_files):
        file_path = os.path.join(KNOWLEDGE_DIR, filename)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(content)
                
        lore_documents.append(content)
        lore_metadata.append({"source": filename, "title": title})
        lore_ids.append(f"doc_{i}")
    
    # Add documents to the vector database
    fantasy_collection.add(
        documents=lore_documents,
        metadatas=lore_metadata,
        ids=lore_ids
    )
    
    print(f"Initialized RAG with {len(lore_documents)} knowledge base documents")
    rag_enabled = True
    
except Exception as e:
    print(f"Failed to initialize RAG components: {e}")
    rag_enabled = False

# Replicate API configuration
REPLICATE_API_TOKEN =  os.getenv("REPLICATE_API_TOKEN")

# Image generation config
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add a static directory for error images
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Create a simple error image if it doesn't exist
error_image_path = os.path.join(STATIC_DIR, "error.png")
if not os.path.exists(error_image_path):
    try:
        # Create a simple error image using PIL
        from PIL import Image, ImageDraw, ImageFont
        error_img = Image.new('RGB', (512, 512), color=(150, 0, 0))
        draw = ImageDraw.Draw(error_img)
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()
        draw.text((100, 200), "Image Generation Error", fill=(255, 255, 255), font=font)
        draw.text((80, 250), "Check API keys or server logs", fill=(255, 255, 255), font=font)
        error_img.save(error_image_path)
        print(f"Created error image at {error_image_path}")
    except Exception as e:
        print(f"Failed to create error image: {e}")

# Story generation caching
STORY_CACHE = {}
CACHE_TTL = 3600  # 1 hour cache lifetime

# Ollama API configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:4b-q4_0")  # Use quantized model for faster performance

# HuggingFace API configuration
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Local image generation config
LOCAL_IMAGE_GENERATION_AVAILABLE = False  # Local generation is disabled

# Add a static directory for error images
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Create a simple error image if it doesn't exist
error_image_path = os.path.join(STATIC_DIR, "error.png")
if not os.path.exists(error_image_path):
    try:
        # Create a simple error image
        error_img = Image.new('RGB', (512, 512), color=(150, 0, 0))
        draw = ImageDraw.Draw(error_img)
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()
        draw.text((100, 200), "Image Generation Error", fill=(255, 255, 255), font=font)
        draw.text((80, 250), "Check API keys or server logs", fill=(255, 255, 255), font=font)
        error_img.save(error_image_path)
        print(f"Created error image at {error_image_path}")
    except Exception as e:
        print(f"Failed to create error image: {e}")

# Ollama availability flag
OLLAMA_AVAILABLE = True

# Story generation functions and utilities
async def generate_with_llamacpp(prompt):
    """Generate text using llama.cpp"""
    global LLAMA_MODEL, LLAMA_MODEL_INITIALIZED
    
    if not LLAMA_CPP_AVAILABLE or not LLAMA_MODEL_INITIALIZED:
        print("llama.cpp not available or not initialized")
        return None
        
    try:
        print("Generating with llama.cpp model")
        
        # Create a system prompt wrapper for better instruction following
        full_prompt = f"""<|system|>
You are a creative fantasy storyteller creating an engaging adventure narrative.
</|system|>

<|user|>
{prompt}
</|user|>

<|assistant|>"""
        
        # Generate text with the model
        output = LLAMA_MODEL.create_completion(
            prompt=full_prompt,
            max_tokens=600,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["<|user|>", "<|system|>", "USER:", "ASSISTANT:"]
        )
        
        # Extract generated text
        if output and "choices" in output and len(output["choices"]) > 0:
            story_text = output["choices"][0]["text"].strip()
            
            # Clean up the response
            story_text = story_text.strip()
            
            # Remove any potential instruction-like text
            blacklist = [
                "write only the story",
                "do not include instructions",
                "as a creative fantasy storyteller",
                "here's the continuation",
                "story text:",
                "story continuation:",
                "here is the story",
                "here's a story"
            ]
            
            for phrase in blacklist:
                if story_text.lower().startswith(phrase):
                    parts = story_text.split("\n", 1)
                    if len(parts) > 1:
                        story_text = parts[1].strip()
            
            # Detect invalid responses that aren't actually story content
            invalid_patterns = [
                "this is a guest",
                "is the author of",
                "published in",
                "this article",
                "this story is based on",
                "written by",
                "fantasy-fiction",
                "magazine",
                "kindle edition",
                "novel was published",
                "guest article",
                "amazon.com",
                "copyright",
                "all rights reserved",
                "fictional",
                "is a retired"
            ]
            
            # Check if the response contains any invalid patterns
            for pattern in invalid_patterns:
                if pattern.lower() in story_text.lower():
                    print(f"Invalid response detected: contains '{pattern}'")
                    return None
            
            # Final length check - if too short, it's probably not a good response
            if len(story_text) < 100:
                print(f"Response too short: {len(story_text)} chars")
                return None
                
            print(f"llama.cpp generated {len(story_text)} chars")
            return story_text
        else:
            print("No valid output from llama.cpp")
            return None
    except Exception as e:
        print(f"Error generating with llama.cpp: {e}")
        traceback.print_exc()
        return None

# Update the generate_with_distilgpt2 function to directly use the local model
def generate_with_distilgpt2(prompt, max_tokens=MAX_TOKENS):
    """Generate story text using local HuggingFace model"""
    try:
        print(f"Generating story with HuggingFace model...")
        
        # Extract character name from the prompt
        character_name_match = re.search(r"about ([A-Za-z]+) in third person", prompt)
        character_name = character_name_match.group(1) if character_name_match else "Hero"
        
        # Generate text using the pipeline
        outputs = story_generator(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            top_k=50,
            num_return_sequences=1,
            return_full_text=False
        )
        
        if outputs and len(outputs) > 0 and outputs[0].get('generated_text'):
            # Get the output text
            story_text = outputs[0]['generated_text'].strip()
            
            # Clean up the text
            story_text = story_text.replace('\n\n', '\n').strip()
            
            # Convert any first-person pronouns to third person
            story_text = story_text.replace(" I ", f" {character_name} ")
            story_text = story_text.replace(" my ", " their ")
            story_text = story_text.replace(" me ", f" {character_name} ")
            story_text = story_text.replace(" am ", " is ")
            
            # Remove any meta text
            blacklist = [
                "write only the story",
                "do not include instructions",
                "as a creative fantasy storyteller",
                "here's the continuation",
                "write a fantasy story",
                "tell a fantasy story"
            ]
            
            for phrase in blacklist:
                if story_text.lower().startswith(phrase):
                    parts = story_text.split("\n", 1)
                    if len(parts) > 1:
                        story_text = parts[1].strip()
            
            # Make sure story is properly formatted as a single paragraph
            sentences = re.split(r'(?<=[.!?])\s+', story_text)
            if sentences:
                # Capitalize first sentence
                if sentences[0] and len(sentences[0]) > 0:
                    sentences[0] = sentences[0][0].upper() + sentences[0][1:] if len(sentences[0]) > 1 else sentences[0].upper()
                
                # Recombine into a clean paragraph
                story_text = " ".join(sentences)
            
            print(f"Generated story: {len(story_text)} chars")
            return story_text
        
        print("No valid output from HuggingFace model")
        return None
    
    except Exception as e:
        print(f"Error generating with HuggingFace model: {e}")
        return None

# Update generate_with_fallbacks to prioritize distilgpt2
async def generate_with_fallbacks(prompt, character_name):
    """Generate story text using available models with fallbacks"""
    story_text = None
    generation_method = None
    
    # Try distilgpt2 first (local model, always available)
    story_text = generate_with_distilgpt2(prompt)
    if story_text and len(story_text) > 20:  # Even a very short story should be usable
        generation_method = "distilgpt2"
        print(f"Generated with distilgpt2: {len(story_text)} chars")
        return story_text, generation_method
        
    # If distilgpt2 fails, try llama.cpp if available
    if LLAMA_CPP_AVAILABLE and LLAMA_MODEL_INITIALIZED:
        story_text = await generate_with_llamacpp(prompt)
        if story_text and len(story_text) > 100:
            # Apply word limit
            words = story_text.split()
            if len(words) > MAX_STORY_WORDS:
                story_text = ' '.join(words[:MAX_STORY_WORDS]) + "..."
                
            generation_method = "llama_cpp"
            print(f"Generated with llama.cpp: {len(story_text)} chars")
            return story_text, generation_method
    
    # Try Ollama if available
    if OLLAMA_AVAILABLE:
        story_text = await generate_with_ollama(prompt)
        if story_text and len(story_text) > 100:
            # Apply word limit
            words = story_text.split()
            if len(words) > MAX_STORY_WORDS:
                story_text = ' '.join(words[:MAX_STORY_WORDS]) + "..."
                
            generation_method = "ollama"
            print(f"Generated with Ollama: {len(story_text)} chars")
            return story_text, generation_method
    
    # Final fallback to template
    story_text = generate_fallback_story(character_name)
    # Apply word limit to template too
    words = story_text.split()
    if len(words) > MAX_STORY_WORDS:
        story_text = ' '.join(words[:MAX_STORY_WORDS]) + "..."
        
    generation_method = "template"
    print(f"Generated with template: {len(story_text)} chars")
    return story_text, generation_method

def extract_key_phrases(text):
    """Extract key phrases from text to maintain story continuity"""
    if not text:
        return []
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Extract noun phrases (simplified approach)
    phrases = []
    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) >= 3:
            # Look for capitalized words that might be important
            for i, word in enumerate(words):
                if word and word[0].isupper() and len(word) > 3 and i < len(words) - 1:
                    # Get the phrase (word and next word)
                    phrase = f"{word} {words[i+1]}"
                    if phrase not in phrases:
                        phrases.append(phrase)
    
    # Extract key locations
    locations = []
    location_keywords = ["castle", "forest", "mountain", "cave", "dungeon", "village", 
                         "town", "city", "temple", "shrine", "tower", "inn", "tavern", 
                         "camp", "river", "lake", "ocean", "sea", "desert"]
    
    for keyword in location_keywords:
        if keyword in text.lower():
            # Find the full phrase containing the location
            pattern = r'\b\w+\s+' + keyword + r'\b|\b' + keyword + r'\s+\w+\b|\b' + keyword + r'\b'
            matches = re.findall(pattern, text.lower())
            if matches:
                for match in matches[:2]:  # Limit to 2 matches per location type
                    if match not in locations:
                        locations.append(match)
    
    # Extract key actions or events
    actions = []
    action_keywords = ["fight", "battle", "attack", "defend", "flee", "escape", "hide",
                      "search", "find", "discover", "talk", "speak", "negotiate", "trade",
                      "buy", "sell", "steal", "ambush", "trap", "explore", "climb", "swim"]
    
    for keyword in action_keywords:
        if keyword in text.lower():
            # Find the full phrase containing the action
            left_pattern = r'\b\w+\s+' + keyword + r'\b'
            right_pattern = r'\b' + keyword + r'\s+\w+\b'
            left_matches = re.findall(left_pattern, text.lower())
            right_matches = re.findall(right_pattern, text.lower())
            
            for match in (left_matches + right_matches)[:2]:  # Limit to 2 matches per action
                if match not in actions:
                    actions.append(match)
    
    # Combine all key phrases
    all_phrases = phrases + locations + actions
    
    # Limit to 5 phrases
    return all_phrases[:5]

# Cache story decisions for faster generation
@lru_cache(maxsize=128)
def get_cached_story(scene_key):
    """Return cached story elements if available to speed up generation"""
    if scene_key in STORY_CACHE:
        cache_entry = STORY_CACHE[scene_key]
        if time.time() - cache_entry['timestamp'] < CACHE_TTL:
            return cache_entry['data']
    return None

def store_in_cache(scene_key, data):
    """Store story generation in cache for reuse"""
    STORY_CACHE[scene_key] = {
        'data': data,
        'timestamp': time.time()
    }

# ==================== HELPER FUNCTIONS ====================
def save_choice_context(player_id, choice_text, story_content, consequence=None):
    """Save the player's choice, story context, and consequence for continuity"""
    if player_id not in GAME_STATES:
        GAME_STATES[player_id] = {"choices": [], "contexts": [], "consequences": []}
        
    GAME_STATES[player_id]["choices"].append(choice_text)
    GAME_STATES[player_id]["contexts"].append(story_content)
    
    if consequence:
        GAME_STATES[player_id]["consequences"].append(consequence)
    else:
        # Add a neutral consequence if none provided
        GAME_STATES[player_id]["consequences"].append({"stat": "none", "change": 0, "description": ""})
    
    # Keep only the most recent 5 entries
    max_history = 5
    if len(GAME_STATES[player_id]["choices"]) > max_history:
        GAME_STATES[player_id]["choices"] = GAME_STATES[player_id]["choices"][-max_history:]
    if len(GAME_STATES[player_id]["contexts"]) > max_history:
        GAME_STATES[player_id]["contexts"] = GAME_STATES[player_id]["contexts"][-max_history:]
    if len(GAME_STATES[player_id]["consequences"]) > max_history:
        GAME_STATES[player_id]["consequences"] = GAME_STATES[player_id]["consequences"][-max_history:]

def get_player_context(player_id):
    """Get a player's choice history and context"""
    if player_id not in GAME_STATES:
        return {"choices": [], "contexts": [], "consequences": []}
    return GAME_STATES[player_id]

def get_story_context(character_name):
    """Get the stored context for a character"""
    if character_name not in story_contexts:
        story_contexts[character_name] = []
    return story_contexts[character_name]

def save_story_context(character_name, context):
    """Save context for a character"""
    story_contexts[character_name] = context
    # Trim context if it gets too long (keep last 10 interactions)
    if len(story_contexts[character_name]) > 10:
        story_contexts[character_name] = story_contexts[character_name][-10:]
    print(f"Updated context for {character_name}: {len(context)} entries")

# ======================= ENDPOINTS =======================

def generate_ollama_prompt(character_name, background, context, previous_choice, story_start=False, location=None):
    """Generate a prompt for Ollama that will produce a good story continuation"""
    prompt = f"""You are a creative fantasy storyteller creating an interactive adventure. Write an engaging story segment about {character_name}, who is {background}.

======== IMPORTANT INSTRUCTIONS ========
1. ONLY write fantasy fiction narrative text. 
2. You are NOT writing an article, blog post, book review, or author bio.
3. DO NOT mention authors, publications, or meta-commentary about fantasy fiction.
4. NEVER create fictional authors or discuss writing/publishing.
5. Write in 3rd person, past tense, focusing ONLY on {character_name}'s adventure.
6. Include sensory details, dialogue, and vivid descriptions of the fantasy world.
7. Keep your tone serious and immersive - this is high fantasy, not meta-fiction.

"""
    
    if story_start:
        prompt += f"""This is the beginning of the adventure. Write an engaging opening (2-3 paragraphs) that:
1. Establishes {character_name}'s personality and motivations
2. Sets up an initial scenario in {location or 'a mysterious fantasy location'}
3. Creates atmosphere and tension with sensory details and vivid imagery
4. Ends with a moment requiring a decision

Example style (don't copy this exactly):
"{character_name} tightened the leather straps of his backpack as the first rays of dawn spilled over the hills. The small farming village where he had spent his entire life was now behind him. Ahead lay the ancient forest, its trees twisted like gnarled fingers reaching for the sky. He took a deep breath, feeling the weight of his father's dagger at his hip..."
"""
    else:
        # Add recent context
        if context and len(context) > 0:
            recent_context = context[-1]['content'] if len(context[-1]['content']) < 500 else context[-1]['content'][:500] + "..."
            prompt += f"""
RECENT STORY CONTEXT:
{recent_context}

THE PLAYER CHOSE: {previous_choice}

Continue the story based on this choice. Write 2-3 paragraphs that:
1. Directly follows from the player's choice
2. Advances the plot in an interesting way 
3. Introduces new elements or challenges
4. Ends with a situation requiring a new decision

Example style (don't copy this exactly):
"{character_name} cautiously approached the ancient stone door. The runes carved into its surface glimmered with an unnatural blue light as his fingers traced their outlines. The ground suddenly trembled beneath his feet, and dust rained from the ceiling..."
"""
        else:
            prompt += f"""
The adventure is already underway. Write 2-3 paragraphs that:
1. Establishes {character_name} in an interesting fantasy situation
2. Creates tension or mystery
3. Provides rich sensory details of the fantasy environment
4. Ends with a moment requiring a decision

Example style (don't copy this exactly):
"The tavern fell silent as {character_name} entered. Cloaked figures huddled in corners, their faces obscured by shadows. A grizzled dwarf behind the bar eyed him suspiciously, one hand resting beneath the counter where a weapon surely waited..."
"""
    
    return prompt

async def generate_with_ollama(prompt):
    """Generate text using Ollama API"""
    if not OLLAMA_AVAILABLE:
        print("Ollama not available, cannot generate story")
        return None
        
    try:
        print(f"Generating with Ollama model: {OLLAMA_MODEL}")
        
        # Prepare the request payload
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 500,
            }
        }
        
        # Make the request to Ollama API
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=30) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Ollama API error: {response.status}, {error_text}")
                    return None
                    
                data = await response.json()
                story_text = data.get('response', '')
                
                # Clean up the response
                story_text = story_text.strip()
                
                # Detect invalid responses that aren't actually story content
                invalid_patterns = [
                    "this is a guest",
                    "is the author of",
                    "published in",
                    "this article",
                    "this story is based on",
                    "written by",
                    "fantasy-fiction",
                    "magazine",
                    "kindle edition",
                    "novel was published",
                    "guest article",
                    "amazon.com",
                    "copyright",
                    "all rights reserved",
                    "fictional",
                    "is a retired"
                ]
                
                # Check if the response contains any invalid patterns
                for pattern in invalid_patterns:
                    if pattern.lower() in story_text.lower():
                        print(f"Invalid response detected: contains '{pattern}'")
                        return None
                
                # Remove any potential instruction-like text
                blacklist = [
                    "write only the story",
                    "do not include instructions",
                    "as a creative fantasy storyteller",
                    "here's the continuation",
                    "story text:",
                    "story continuation:",
                    "here is the story",
                    "here's a story"
                ]
                
                for phrase in blacklist:
                    if story_text.lower().startswith(phrase):
                        parts = story_text.split("\n", 1)
                        if len(parts) > 1:
                            story_text = parts[1].strip()
                
                # Final length check - if too short, it's probably not a good response
                if len(story_text) < 100:
                    print(f"Response too short: {len(story_text)} chars")
                    return None
                    
                return story_text
    except Exception as e:
        print(f"Error calling Ollama API: {e}")
        return None

def generate_contextual_options(story_text, character_name):
    """Generate contextual options based on the story content"""
    # Default options if we can't generate contextual ones
    default_options = [
        f"Continue exploring cautiously",
        f"Look for another path",
        f"Rest and recover strength"
    ]
    
    try:
        # Extract key elements from the story
        key_phrases = extract_key_phrases(story_text)
        
        if not story_text or not key_phrases:
            return default_options
            
        # Look for specific story elements to create more relevant options
        options = []
        story_lower = story_text.lower()
        
        # Check for character interactions
        character_interactions = []
        interaction_keywords = ["talk", "spoke", "said", "asked", "replied", "shouted", "whispered", "called"]
        for keyword in interaction_keywords:
            if keyword in story_lower:
                character_interactions.append("Speak with them")
                character_interactions.append("Ask about their knowledge")
                break
                
        # Check for items or objects
        item_keywords = ["sword", "staff", "potion", "scroll", "book", "map", "chest", "door", "gate", "key"]
        item_actions = []
        for keyword in item_keywords:
            if keyword in story_lower:
                item_actions.append(f"Examine the {keyword}")
                if keyword in ["chest", "door", "gate"]:
                    item_actions.append(f"Try to open the {keyword}")
                if keyword in ["sword", "staff"]:
                    item_actions.append(f"Take the {keyword}")
                break
                
        # Check for location-based actions
        location_keywords = {
            "cave": ["Explore deeper", "Look for hidden passages"],
            "forest": ["Move quietly through the trees", "Climb a tree for a better view"],
            "mountain": ["Climb higher", "Look for a safe path down"],
            "castle": ["Search for the throne room", "Look for secret passages"],
            "ruins": ["Search for ancient artifacts", "Decipher the wall carvings"],
            "village": ["Speak with the villagers", "Visit the local tavern"],
            "river": ["Follow the river downstream", "Try to cross the river"],
            "tower": ["Climb to the top", "Look for a way in"]
        }
        
        location_actions = []
        for location, actions in location_keywords.items():
            if location in story_lower:
                location_actions.extend(actions)
                break

        # Check for danger or combat situations
        danger_keywords = ["monster", "creature", "beast", "enemy", "trap", "danger", "attack"]
        danger_actions = []
        for keyword in danger_keywords:
            if keyword in story_lower:
                danger_actions = ["Prepare for combat", "Look for a way to escape", "Hide and observe"]
                break
                
        # Combine all potential actions
        all_actions = character_interactions + item_actions + location_actions + danger_actions
        
        # Add actions based on key phrases if we still need more options
        for phrase in key_phrases:
            phrase_lower = phrase.lower()
            if "path" in phrase_lower or "road" in phrase_lower:
                all_actions.append(f"Follow the {phrase}")
            elif any(creature in phrase_lower for creature in ["dragon", "troll", "ogre", "goblin", "wolf"]):
                all_actions.append(f"Approach the {phrase} cautiously")
                all_actions.append(f"Hide from the {phrase}")
            else:
                all_actions.append(f"Investigate the {phrase}")
        
        # Deduplicate and get the most relevant options
        seen = set()
        options = [action for action in all_actions if action not in seen and not seen.add(action)]
        
        # Ensure we have exactly 3 options
        if len(options) > 3:
            options = options[:3]
        
        # If we couldn't generate enough options, add some defaults
        while len(options) < 3:
            for opt in default_options:
                if opt not in options:
                    options.append(opt)
                    if len(options) == 3:
                        break
        
        return options
        
    except Exception as e:
        print(f"Error generating options: {e}")
        return default_options

def generate_story_prompt(character, context=None, key_elements=None, previous_choice=None, story_start=False, location=None):
    """Generate a prompt for story continuation with improved prompt tuning and context usage"""
    character_name = character.name if character else "Hero"
    background = character.background if character else "An adventurer seeking fortune and glory"
    
    # Create a much more constrained prompt with explicit instructions
    prompt = f"""Generate a SHORT FANTASY STORY PARAGRAPH about {character_name}, who is {background}.

STRICT REQUIREMENTS:
1. Write exactly ONE paragraph (3-5 sentences).
2. Focus ONLY on {character_name}'s immediate actions and surroundings.
3. Use third-person perspective (he/she/they).
4. Include vivid sensory details (sights, sounds, smells).
5. End with a moment of tension or decision.
6. DO NOT introduce the story with phrases like "This is a story about..." or "Once upon a time..."
7. Avoid all meta-commentary about the story itself.
"""
    
    # Build up the context with more specificity for continuity
    if story_start:
        prompt += f"\nThis is the beginning of {character_name}'s adventure in {location or 'a mysterious realm'}.\n"
    elif previous_choice and context:
        # Make the previous choice more prominent in the prompt
        prompt += f"\nPrevious scene: {context}\n\n{character_name} has chosen to: {previous_choice.lower()}\n\nContinue the story directly from this choice, showing what happens as a result.\n"
    elif previous_choice:
        prompt += f"\n{character_name} has chosen to: {previous_choice.lower()}\n\nShow the immediate consequences of this choice.\n"
    elif context:
        prompt += f"\nPrevious event: {context}\nContinue the story from this point.\n"
    
    prompt += "\nSTORY TEXT:"
    
    return prompt

async def generate_with_huggingface(prompt, character_name, max_tokens=MAX_TOKENS):
    """Generate story text using HuggingFace model with significantly improved validation"""
    try:
        print(f"Generating story with HuggingFace model for {character_name}...")
        
        # Generate text using the pipeline with adjusted parameters for better results
        outputs = story_generator(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,    # Higher temperature for more creativity
            top_p=0.92,         # Slightly higher nucleus sampling
            top_k=80,           # Allow more token diversity
            repetition_penalty=1.2,  # Explicitly penalize repetition
            num_return_sequences=1,
            return_full_text=False
        )
        
        if outputs and len(outputs) > 0 and outputs[0].get('generated_text'):
            # Get the output text
            story_text = outputs[0]['generated_text'].strip()
            
            # Clean up the text
            story_text = story_text.replace('\n\n', ' ').replace('\n', ' ').strip()
            
            # VALIDATION CHECKS - with less strict repetition check
            
            # 1. Check for excessive repeating phrases (allowing some repetition)
            words = story_text.split()
            if len(words) > 12:
                # Check for repeating 4+ word phrases instead of 3+
                # This allows some natural repetition while catching bad generation
                phrases = []
                for i in range(len(words) - 3):
                    phrase = ' '.join(words[i:i+4]).lower()
                    phrases.append(phrase)
                
                # Count occurrences of each phrase
                phrase_counts = {}
                for phrase in phrases:
                    if phrase in phrase_counts:
                        phrase_counts[phrase] += 1
                    else:
                        phrase_counts[phrase] = 1
                
                # Only reject if we have phrases repeated 3 or more times
                excessive_repetition = any(count >= 3 for count in phrase_counts.values())
                if excessive_repetition:
                    print("Rejected story with excessive repetition")
                    return None
            
            # 2. Check for banned phrases and patterns
            banned_phrases = [
                "this is a story", 
                "the story is", 
                "this story", 
                "the tale", 
                "once upon a time",
                "a boy named",
                "a girl named",
                "the story begins",
                "our story",
                "in this story",
                "based on"
            ]
            
            if any(phrase in story_text.lower() for phrase in banned_phrases):
                print("Rejected story with banned phrases")
                return None
            
            # 3. Check for too many questions or first-person references
            if story_text.count('?') > 1 or " I " in story_text or " my " in story_text:
                print("Rejected story with too many questions or first-person perspective")
                return None
            
            # 4. Extract just the first paragraph if multiple paragraphs
            if "\n\n" in story_text:
                story_text = story_text.split("\n\n")[0].strip()
            
            # 5. Ensure the story is about the character
            if character_name not in story_text:
                print("Rejected story that doesn't mention the character name")
                return None
            
            # 6. Convert any first-person pronouns to third person just in case
            story_text = story_text.replace(" I ", f" {character_name} ")
            story_text = story_text.replace(" My ", f" {character_name}'s ")
            story_text = story_text.replace(" my ", f" {character_name}'s ")
            story_text = story_text.replace(" me ", f" {character_name} ")
            story_text = story_text.replace(" am ", " is ")
            
            # 7. Make sure story is properly formatted as a single paragraph
            sentences = re.split(r'(?<=[.!?])\s+', story_text)
            
            # Filter out instruction-like or meta-text sentences
            filtered_sentences = []
            for sentence in sentences:
                # Skip sentences with instruction-like language
                if re.search(r'\b(write|describing|instructions|guidelines|narrative|focus|prompt|create|tell)\b', sentence.lower()):
                    continue
                    
                # Skip sentences with meta-commentary
                if re.search(r'\b(this story|the story|a story about|this tale|the tale|fiction|based on|inspired by)\b', sentence.lower()):
                    continue
                    
                # Skip very short fragments
                if len(sentence.split()) < 3:
                    continue
                    
                filtered_sentences.append(sentence)
            
            # Ensure we have at least 2 sentences
            if len(filtered_sentences) < 2:
                print("Rejected story with insufficient content after filtering")
                return None
                
            # Capitalize first sentence if needed
            if filtered_sentences[0] and len(filtered_sentences[0]) > 0:
                filtered_sentences[0] = filtered_sentences[0][0].upper() + filtered_sentences[0][1:] if len(filtered_sentences[0]) > 1 else filtered_sentences[0].upper()
            
            # Recombine into a clean paragraph
            story_text = " ".join(filtered_sentences)
            
            # Apply word limit - ensure it ends with a complete sentence
            words = story_text.split()
            if len(words) > MAX_STORY_WORDS:
                story_text = ' '.join(words[:MAX_STORY_WORDS])
                last_period = story_text.rfind('.')
                last_exclamation = story_text.rfind('!')
                last_question = story_text.rfind('?')
                last_boundary = max(last_period, last_exclamation, last_question)
                if last_boundary > len(story_text) * 0.6:
                    story_text = story_text[:last_boundary+1]
 
            # Final verification - story length
            if len(story_text) < 60 or len(story_text.split()) < 15:
                print("Rejected story that's too short")
                return None
                
            print(f"Generated story: {len(story_text)} chars, {len(story_text.split())} words")
            return story_text
        
        print("No valid output from HuggingFace model")
        return None
    
    except Exception as e:
        print(f"Error generating with HuggingFace model: {e}")
        return None

@app.post("/generate-story", response_model=StoryResponse)
async def generate_story(req: StoryRequest):
    """Generate a story segment with options based on the current state"""
    try:
        character_name = req.character_name if req.character_name else req.character.name if req.character else "Hero"
        previous_choice = req.previous_choice if req.previous_choice else ""
        story_start = req.story_start if req.story_start else False
        location = req.location if req.location else None
        
        # Initialize character if not provided
        if not req.character:
            req.character = CharacterModel(name=character_name)
            
        # Initialize stats if not present
        if not req.character.stats:
            req.character.stats = {
                "strength": 10,
                "wisdom": 10,
                "agility": 10,
                "charisma": 10,
                "health": 100,
                "magic": 10
            }
        
        # Get the character's context from RAG or game state
        context = None
        last_options = []
        
        if character_name in GAME_STATES:
            if len(GAME_STATES[character_name]["contexts"]) > 0:
                context = GAME_STATES[character_name]["contexts"][-1]
            if "last_options" in GAME_STATES[character_name]:
                last_options = GAME_STATES[character_name]["last_options"]
        
        # Track and apply stat changes from previous choice
        stat_change_message = ""
        if previous_choice and character_name in GAME_STATES:
            if "last_options_data" in GAME_STATES[character_name] and GAME_STATES[character_name]["last_options_data"]:
                # Find which option was chosen
                for option_data in GAME_STATES[character_name]["last_options_data"]:
                    if option_data["text"] == previous_choice:
                        consequence = option_data["consequence"]
                        stat = consequence["stat"]
                        change = consequence["change"]
                        
                        if stat != "none" and stat in req.character.stats:
                            # Apply the change
                            old_value = req.character.stats[stat]
                            req.character.stats[stat] += change
                            
                            # Record the stat change for display
                            stat_change_message = f"[Your {stat} changed from {old_value} to {req.character.stats[stat]}] {consequence['description']}"
                            print(f"Updated {character_name}'s {stat} by {change} to {req.character.stats[stat]}")
                        break
        
        # Try with a more robust approach - up to 3 attempts with different generation parameters
        story_text = None
        for attempt in range(3):
            try:
                # Create a new prompt for each attempt to add some variation
                prompt = generate_story_prompt(req.character, context, [], previous_choice, story_start, location)
                
                # Adjust generation parameters based on attempt number
                temp = 0.8 if attempt == 0 else (0.7 if attempt == 1 else 0.9)
                
                # Directly modify the generator for this attempt
                story_generator.temperature = temp
                story_generator.repetition_penalty = 1.2 + (attempt * 0.1) # Increase penalty with each attempt
                
                story_text = await generate_with_huggingface(prompt, character_name)
                if story_text:
                    break
                print(f"Story generation attempt {attempt+1} failed, retrying with different parameters...")
            except Exception as e:
                print(f"Error in generation attempt {attempt+1}: {e}")
                continue
        
        # Fall back to template if all attempts fail
        if not story_text:
            print("All generation attempts failed, using fallback")
            story_text = generate_fallback_story(character_name, location, previous_choice)
        
        # Add the stat change message to the story text if applicable
        if stat_change_message:
            story_text = stat_change_message + "\n\n" + story_text
        
        # Generate contextual options with consequences, avoiding repeats from last time
        options_with_consequences = generate_options_with_consequences(
            story_text, 
            character_name, 
            req.character.stats,
            last_options
        )
        
        # Extract just the option text for the response
        options = [choice["text"] for choice in options_with_consequences]
        
        # Create formatted choices with consequences
        formatted_choices = [
            ChoiceModel(
                text=choice["text"],
                consequence=ConsequenceModel(
                    stat=choice["consequence"]["stat"],
                    change=choice["consequence"]["change"],
                    description=choice["consequence"]["description"]
                )
            ) for choice in options_with_consequences
        ]
        
        # Store context in GAME_STATES
        if character_name not in GAME_STATES:
            GAME_STATES[character_name] = {
                "choices": [], 
                "contexts": [],
                "consequences": [],
                "last_options": [],
                "last_options_data": []
            }
        
        # Save the current choice and context
        if previous_choice:
            GAME_STATES[character_name]["choices"].append(previous_choice)
        
        # Always add the current context
        GAME_STATES[character_name]["contexts"].append(story_text)
        
        # Save the current options to avoid repeating them next time
        GAME_STATES[character_name]["last_options"] = options
        GAME_STATES[character_name]["last_options_data"] = options_with_consequences
        
        # Trim context if it gets too long
        max_history = 5
        if len(GAME_STATES[character_name]["contexts"]) > max_history:
            GAME_STATES[character_name]["contexts"] = GAME_STATES[character_name]["contexts"][-max_history:]
        if len(GAME_STATES[character_name]["choices"]) > max_history:
            GAME_STATES[character_name]["choices"] = GAME_STATES[character_name]["choices"][-max_history:]
        if "consequences" in GAME_STATES[character_name] and len(GAME_STATES[character_name]["consequences"]) > max_history:
            GAME_STATES[character_name]["consequences"] = GAME_STATES[character_name]["consequences"][-max_history:]
        
        # Store in RAG if available
        if rag_enabled and (story_start or previous_choice):
            try:
                # Create a document for the story segment
                doc_id = f"story_{character_name}_{int(time.time())}"
                
                # Include the previous choice in the document metadata
                metadata = {
                    "character": character_name,
                    "type": "story_segment",
                    "choice": previous_choice if previous_choice else "story_start",
                    "background": req.character.background if req.character else "adventurer"
                }
                
                # Store the story text AND the choice that led to it for better context
                document_text = f"Choice: {previous_choice}\nStory: {story_text}" if previous_choice else story_text
                
                fantasy_collection.add(
                    documents=[document_text],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                print(f"Stored story in RAG with ID: {doc_id}")
                
                # If we have a previous choice, try to use RAG to find relevant context
                if previous_choice and not story_start:
                    try:
                        # Query for most relevant previous contexts
                        results = fantasy_collection.query(
                            query_texts=[previous_choice],
                            n_results=2,
                            where={"character": character_name}
                        )
                        
                        if results and results.get('documents') and len(results['documents'][0]) > 0:
                            print(f"Found relevant previous context in RAG")
                    except Exception as rag_error:
                        print(f"Error searching RAG: {rag_error}")
                        
            except Exception as e:
                print(f"Error storing story in RAG: {e}")
        
        return StoryResponse(
            content=story_text,
            choices=formatted_choices,
            character=req.character,
            image_prompt=f"Fantasy adventure scene with {character_name}",
            story=story_text,
            options=options
        )
            
    except Exception as e:
        print(f"Story generation error: {str(e)}")
        traceback.print_exc()
        
        # Create default choices for error case
        default_options = ["Start a new adventure", "Try again", "Change your character"]
        formatted_choices = [
            ChoiceModel(text=option, consequence=ConsequenceModel()) 
            for option in default_options
        ]
        
        return StoryResponse(
            content="Something went wrong with the story. The words seemed to fade away...",
            choices=formatted_choices,
            character=req.character if req.character else None,
            image_prompt="Fantasy scene",
            story="Something went wrong with the story. The words seemed to fade away...",
            options=default_options
        )

def generate_options_with_consequences(story_text, character_name, character_stats=None, previous_options=None):
    """Generate options with consequences based on story content, avoiding repetition"""
    
    # Default stats if none provided
    if not character_stats:
        character_stats = {
            "strength": 10,
            "wisdom": 10,
            "agility": 10,
            "charisma": 10,
            "health": 100,
            "magic": 10
        }
    
    # Initialize previous_options if not provided
    if previous_options is None:
        previous_options = []
    
    # Try to detect the situation type from the story
    situation = "unknown"
    
    # Additional situation types for more variety
    situation_keywords = {
        "combat": ["attack", "fight", "battle", "enemy", "threat", "danger", "weapon", "defend", "sword", "dagger", "shield"],
        "exploration": ["path", "passage", "cave", "forest", "ruins", "temple", "castle", "dungeon", "discover", "explore", "search"],
        "social": ["speak", "talk", "conversation", "village", "town", "city", "tavern", "inn", "person", "people", "crowd"],
        "mystery": ["mystery", "puzzle", "strange", "unusual", "curious", "investigate", "secret", "hidden", "unknown", "suspicious"],
        "danger": ["trap", "snare", "ambush", "lurking", "stalking", "watching", "following", "dark", "ominous", "foreboding"],
        "magic": ["spell", "magic", "enchant", "glow", "aura", "mystic", "arcane", "ritual", "rune", "symbol", "power"],
        "nature": ["tree", "river", "animal", "plant", "flower", "grass", "wind", "rain", "stream", "bird", "beast"],
        "treasure": ["gold", "silver", "gem", "jewel", "treasure", "coin", "wealth", "rich", "valuable", "chest", "hoard"],
        "stealth": ["hide", "sneak", "quiet", "shadow", "silent", "disguise", "conceal", "cloak", "mask", "unnoticed", "slip"],
        "puzzle": ["riddle", "puzzle", "contraption", "mechanism", "lock", "key", "solution", "solve", "answer", "clue", "hint"]
    }
    
    story_lower = story_text.lower()
    situation_scores = {sit: 0 for sit in situation_keywords}
    
    for sit, keywords in situation_keywords.items():
        for keyword in keywords:
            if keyword in story_lower:
                situation_scores[sit] += 1
    
    # Get the top two most likely situations for more varied options
    sorted_situations = sorted(situation_scores.items(), key=lambda x: x[1], reverse=True)
    primary_situation = sorted_situations[0][0] if sorted_situations[0][1] > 0 else "unknown"
    secondary_situation = sorted_situations[1][0] if len(sorted_situations) > 1 and sorted_situations[1][1] > 0 else "exploration"
    
    # Expanded option sets for each situation type
    all_situation_options = {
        "combat": [
            {"text": "Ready your weapon", "consequence": {"stat": "strength", "change": 1, "description": "Combat preparedness improved"}},
            {"text": "Look for a defensive position", "consequence": {"stat": "health", "change": 2, "description": "Found safer ground"}},
            {"text": "Attempt to negotiate", "consequence": {"stat": "charisma", "change": 1, "description": "Diplomatic skills improved"}},
            {"text": "Call for assistance", "consequence": {"stat": "charisma", "change": -1, "description": "No one came to help"}},
            {"text": "Prepare a spell", "consequence": {"stat": "magic", "change": 2, "description": "Magical focus sharpened"}},
            {"text": "Assess your opponent's weakness", "consequence": {"stat": "wisdom", "change": 1, "description": "Tactical insight gained"}},
            {"text": "Draw your weapon and stand firm", "consequence": {"stat": "strength", "change": 1, "description": "Combat stance improved"}},
            {"text": "Attempt to flank your enemy", "consequence": {"stat": "agility", "change": 2, "description": "Movement tactics refined"}},
            {"text": "Intimidate your opponent", "consequence": {"stat": "charisma", "change": 2, "description": "Successfully intimidated foe"}},
            {"text": "Use surroundings to your advantage", "consequence": {"stat": "wisdom", "change": 2, "description": "Used terrain tactically"}},
            {"text": "Strike first, before they're ready", "consequence": {"stat": "strength", "change": 1, "description": "Seized the initiative"}},
            {"text": "Look for an opportunity to disarm them", "consequence": {"stat": "agility", "change": 1, "description": "Disarming technique practiced"}}
        ],
        "exploration": [
            {"text": "Investigate further", "consequence": {"stat": "wisdom", "change": 1, "description": "Knowledge expanded"}},
            {"text": "Search for hidden passages", "consequence": {"stat": "agility", "change": -1, "description": "Exhausted from searching"}},
            {"text": "Examine your surroundings carefully", "consequence": {"stat": "wisdom", "change": 2, "description": "Noticed important details"}},
            {"text": "Look for tracks or signs of others", "consequence": {"stat": "wisdom", "change": 1, "description": "Tracking skills improved"}},
            {"text": "Climb to a better vantage point", "consequence": {"stat": "agility", "change": 1, "description": "Climbing ability enhanced"}},
            {"text": "Map your surroundings", "consequence": {"stat": "wisdom", "change": 1, "description": "Area knowledge improved"}},
            {"text": "Check for traps before proceeding", "consequence": {"stat": "health", "change": 2, "description": "Avoided potential danger"}},
            {"text": "Follow the most promising path", "consequence": {"stat": "agility", "change": 1, "description": "Found an efficient route"}},
            {"text": "Mark your trail to avoid getting lost", "consequence": {"stat": "wisdom", "change": 1, "description": "Navigation improved"}},
            {"text": "Test the ground before proceeding", "consequence": {"stat": "health", "change": 1, "description": "Prevented potential injury"}},
            {"text": "Look for resources you can use", "consequence": {"stat": "wisdom", "change": 2, "description": "Found useful materials"}},
            {"text": "Scout ahead before committing", "consequence": {"stat": "agility", "change": 1, "description": "Reconnaissance skills improved"}}
        ],
        "social": [
            {"text": "Approach and introduce yourself", "consequence": {"stat": "charisma", "change": 1, "description": "Social skills improved"}},
            {"text": "Listen to conversations around you", "consequence": {"stat": "wisdom", "change": 1, "description": "Gathered useful information"}},
            {"text": "Ask about local rumors", "consequence": {"stat": "charisma", "change": 1, "description": "Built rapport with locals"}},
            {"text": "Offer your assistance", "consequence": {"stat": "charisma", "change": 2, "description": "Reputation enhanced"}},
            {"text": "Share your story", "consequence": {"stat": "charisma", "change": -1, "description": "Story was not well-received"}},
            {"text": "Buy a round of drinks", "consequence": {"stat": "charisma", "change": 2, "description": "Made new friends"}},
            {"text": "Seek information about your quest", "consequence": {"stat": "wisdom", "change": 1, "description": "Gained quest knowledge"}},
            {"text": "Look for someone in charge", "consequence": {"stat": "charisma", "change": 1, "description": "Connected with authority"}},
            {"text": "Find common ground with locals", "consequence": {"stat": "charisma", "change": 2, "description": "Built trust with community"}},
            {"text": "Observe social customs before engaging", "consequence": {"stat": "wisdom", "change": 1, "description": "Cultural awareness improved"}},
            {"text": "Show respect to local traditions", "consequence": {"stat": "charisma", "change": 2, "description": "Earned locals' respect"}},
            {"text": "Mediate a local dispute", "consequence": {"stat": "charisma", "change": 2, "description": "Diplomatic reputation grew"}}
        ],
        "mystery": [
            {"text": "Investigate the strange phenomenon", "consequence": {"stat": "magic", "change": 1, "description": "Magical insight increased"}},
            {"text": "Look for clues", "consequence": {"stat": "wisdom", "change": 1, "description": "Deduction skills improved"}},
            {"text": "Try to recall relevant knowledge", "consequence": {"stat": "wisdom", "change": 2, "description": "Memory strengthened"}},
            {"text": "Test your theory carefully", "consequence": {"stat": "magic", "change": -1, "description": "Theory backfired slightly"}},
            {"text": "Document what you've found", "consequence": {"stat": "wisdom", "change": 1, "description": "Improved analytical skills"}},
            {"text": "Search for patterns or symbols", "consequence": {"stat": "wisdom", "change": 1, "description": "Pattern recognition enhanced"}},
            {"text": "Use magic to reveal hidden details", "consequence": {"stat": "magic", "change": 2, "description": "Magical perception increased"}},
            {"text": "Compare your findings with known lore", "consequence": {"stat": "wisdom", "change": 2, "description": "Lore knowledge expanded"}},
            {"text": "Recreate the conditions that caused it", "consequence": {"stat": "magic", "change": 2, "description": "Experimental method improved"}},
            {"text": "Seek historical records about this place", "consequence": {"stat": "wisdom", "change": 2, "description": "Historical knowledge gained"}},
            {"text": "Test if the phenomenon responds to touch", "consequence": {"stat": "magic", "change": -2, "description": "Magical feedback experienced"}},
            {"text": "Look for similar occurrences nearby", "consequence": {"stat": "wisdom", "change": 1, "description": "Pattern recognition improved"}}
        ],
        "danger": [
            {"text": "Move silently and stay hidden", "consequence": {"stat": "agility", "change": 1, "description": "Stealth improved"}},
            {"text": "Ready an escape route", "consequence": {"stat": "agility", "change": 1, "description": "Escape planning improved"}},
            {"text": "Set up a defensive position", "consequence": {"stat": "health", "change": 1, "description": "Found safer ground"}},
            {"text": "Prepare for an ambush", "consequence": {"stat": "strength", "change": 1, "description": "Combat readiness improved"}},
            {"text": "Look for alternative paths", "consequence": {"stat": "agility", "change": 1, "description": "Found a safer route"}},
            {"text": "Try to identify the source of danger", "consequence": {"stat": "wisdom", "change": 1, "description": "Threat assessment improved"}},
            {"text": "Signal for help", "consequence": {"stat": "charisma", "change": -1, "description": "Signal went unanswered"}},
            {"text": "Create a distraction", "consequence": {"stat": "wisdom", "change": 1, "description": "Tactical thinking improved"}},
            {"text": "Set traps of your own", "consequence": {"stat": "wisdom", "change": 2, "description": "Trap-setting skills improved"}},
            {"text": "Lure the threat away from you", "consequence": {"stat": "agility", "change": 2, "description": "Evasion tactics practiced"}},
            {"text": "Use the environment for cover", "consequence": {"stat": "health", "change": 1, "description": "Found good protection"}},
            {"text": "Prepare a quick escape plan", "consequence": {"stat": "agility", "change": 1, "description": "Emergency planning improved"}}
        ],
        "magic": [
            {"text": "Study the magical aura", "consequence": {"stat": "magic", "change": 2, "description": "Magical sensitivity increased"}},
            {"text": "Attempt to decipher the runes", "consequence": {"stat": "wisdom", "change": 1, "description": "Rune knowledge expanded"}},
            {"text": "Channel your magical energy", "consequence": {"stat": "magic", "change": 1, "description": "Magical control improved"}},
            {"text": "Cast a protective spell", "consequence": {"stat": "magic", "change": 1, "description": "Defensive magic strengthened"}},
            {"text": "Test the magical properties", "consequence": {"stat": "magic", "change": -1, "description": "Magical experiment backfired"}},
            {"text": "Look for a magical focus object", "consequence": {"stat": "magic", "change": 1, "description": "Found a magical catalyst"}},
            {"text": "Recall ancient magical teachings", "consequence": {"stat": "wisdom", "change": 1, "description": "Magical theory knowledge improved"}},
            {"text": "Draw power from the surroundings", "consequence": {"stat": "magic", "change": 2, "description": "Environmental attunement increased"}},
            {"text": "Counter the magical energy", "consequence": {"stat": "magic", "change": 2, "description": "Magical resistance improved"}},
            {"text": "Harmonize your aura with the magic", "consequence": {"stat": "magic", "change": 2, "description": "Magical attunement increased"}},
            {"text": "Record the magical patterns you observe", "consequence": {"stat": "wisdom", "change": 2, "description": "Magical documentation improved"}},
            {"text": "Attempt to absorb some of the magic", "consequence": {"stat": "magic", "change": -2, "description": "Magical overload experienced"}}
        ],
        "nature": [
            {"text": "Follow animal tracks", "consequence": {"stat": "wisdom", "change": 1, "description": "Tracking skills improved"}},
            {"text": "Identify edible plants", "consequence": {"stat": "wisdom", "change": 2, "description": "Foraging knowledge increased"}},
            {"text": "Listen to the sounds of nature", "consequence": {"stat": "wisdom", "change": 1, "description": "Environmental awareness enhanced"}},
            {"text": "Find a safe spot to observe wildlife", "consequence": {"stat": "agility", "change": 1, "description": "Stealth in nature improved"}},
            {"text": "Collect useful herbs", "consequence": {"stat": "health", "change": 1, "description": "Herbal knowledge expanded"}},
            {"text": "Look for signs of water nearby", "consequence": {"stat": "wisdom", "change": 1, "description": "Survival skills improved"}},
            {"text": "Set up a temporary shelter", "consequence": {"stat": "health", "change": 1, "description": "Outdoor survival skills practiced"}},
            {"text": "Study the patterns of plant growth", "consequence": {"stat": "wisdom", "change": 1, "description": "Botanical knowledge improved"}}
        ],
        "treasure": [
            {"text": "Examine the valuables carefully", "consequence": {"stat": "wisdom", "change": 1, "description": "Appraisal skills improved"}},
            {"text": "Check for traps before touching anything", "consequence": {"stat": "health", "change": 2, "description": "Avoided potential trap"}},
            {"text": "Look for secret compartments", "consequence": {"stat": "wisdom", "change": 1, "description": "Detection skills enhanced"}},
            {"text": "Test if any items are magical", "consequence": {"stat": "magic", "change": 1, "description": "Magical sensitivity increased"}},
            {"text": "Select the most valuable items", "consequence": {"stat": "wisdom", "change": 1, "description": "Value assessment improved"}},
            {"text": "Properly store the treasures", "consequence": {"stat": "agility", "change": 1, "description": "Item handling skills improved"}},
            {"text": "Look for ownership marks", "consequence": {"stat": "wisdom", "change": 1, "description": "Attention to detail improved"}},
            {"text": "Create an inventory of what you've found", "consequence": {"stat": "wisdom", "change": 1, "description": "Organization skills practiced"}}
        ],
        "stealth": [
            {"text": "Move silently through shadows", "consequence": {"stat": "agility", "change": 2, "description": "Stealth movement improved"}},
            {"text": "Find a hiding spot with good visibility", "consequence": {"stat": "agility", "change": 1, "description": "Concealment tactics improved"}},
            {"text": "Time your movements carefully", "consequence": {"stat": "agility", "change": 1, "description": "Patience and timing practiced"}},
            {"text": "Create a diversion to mask your presence", "consequence": {"stat": "wisdom", "change": 1, "description": "Distraction tactics improved"}},
            {"text": "Disguise your appearance", "consequence": {"stat": "charisma", "change": 1, "description": "Disguise skills practiced"}},
            {"text": "Listen for signs of detection", "consequence": {"stat": "wisdom", "change": 1, "description": "Auditory awareness increased"}},
            {"text": "Disable any alerting mechanisms", "consequence": {"stat": "agility", "change": 1, "description": "Mechanical skill improved"}},
            {"text": "Use natural sounds to cover your movements", "consequence": {"stat": "agility", "change": 1, "description": "Sound masking technique improved"}}
        ],
        "puzzle": [
            {"text": "Study the mechanism carefully", "consequence": {"stat": "wisdom", "change": 2, "description": "Analytical skills improved"}},
            {"text": "Look for clues in the surroundings", "consequence": {"stat": "wisdom", "change": 1, "description": "Environmental awareness enhanced"}},
            {"text": "Test different combinations", "consequence": {"stat": "wisdom", "change": 1, "description": "Problem-solving improved"}},
            {"text": "Check for hidden instructions", "consequence": {"stat": "wisdom", "change": 1, "description": "Detail observation skills increased"}},
            {"text": "Apply logical reasoning", "consequence": {"stat": "wisdom", "change": 2, "description": "Logical thinking strengthened"}},
            {"text": "See if magic affects the puzzle", "consequence": {"stat": "magic", "change": 1, "description": "Magical problem-solving practiced"}},
            {"text": "Look for patterns in the design", "consequence": {"stat": "wisdom", "change": 1, "description": "Pattern recognition improved"}},
            {"text": "Compare to puzzles you've solved before", "consequence": {"stat": "wisdom", "change": 1, "description": "Experience applied successfully"}}
        ],
        "unknown": [
            {"text": "Assess the situation carefully", "consequence": {"stat": "wisdom", "change": 1, "description": "Observational skills improved"}},
            {"text": "Proceed with caution", "consequence": {"stat": "health", "change": 1, "description": "Avoided potential dangers"}},
            {"text": "Consider your options", "consequence": {"stat": "wisdom", "change": 1, "description": "Decision-making improved"}},
            {"text": "Take a moment to plan", "consequence": {"stat": "wisdom", "change": 1, "description": "Strategic thinking enhanced"}},
            {"text": "Look for any advantages", "consequence": {"stat": "wisdom", "change": 1, "description": "Situational awareness improved"}},
            {"text": "Prepare for the unexpected", "consequence": {"stat": "health", "change": 1, "description": "Readiness increased"}},
            {"text": "Trust your instincts", "consequence": {"stat": "agility", "change": 1, "description": "Intuitive response improved"}},
            {"text": "Recall your training", "consequence": {"stat": "strength", "change": 1, "description": "Applied training knowledge"}},
            {"text": "Gather your thoughts", "consequence": {"stat": "wisdom", "change": 1, "description": "Mental focus improved"}},
            {"text": "Test the safety of your surroundings", "consequence": {"stat": "health", "change": 1, "description": "Caution practiced successfully"}},
            {"text": "Search for anything unusual", "consequence": {"stat": "wisdom", "change": 1, "description": "Attention to detail enhanced"}},
            {"text": "Ready yourself for action", "consequence": {"stat": "strength", "change": 1, "description": "Preparedness improved"}}
        ]
    }
    
    # Extract key entities from the story
    entities = []
    
    # Look for common fantasy objects/locations
    keywords = ["cave", "forest", "castle", "tower", "ruins", "temple", "mountain", "village", 
                "sword", "staff", "wand", "scroll", "potion", "door", "gate", "bridge", 
                "chest", "dragon", "creature", "beast", "stranger", "wizard", "warrior",
                "tomb", "crypt", "shrine", "altar", "statue", "monument", "library",
                "tavern", "market", "camp", "settlement", "fortress", "outpost"]
    
    for keyword in keywords:
        if keyword in story_lower:
            entities.append(keyword)
    
    # Create options list
    final_options = []
    
    # Add entity-specific options if we found entities
    entity_options = []
    if entities:
        random.shuffle(entities)
        for entity in entities[:3]:  # Try more entities
            if entity in ["cave", "forest", "castle", "tower", "ruins", "temple", "mountain", "village", "tomb", "crypt", "shrine", "library", "tavern", "market", "camp", "settlement", "fortress", "outpost"]:
                entity_options.append({
                    "text": f"Explore the {entity} further",
                    "consequence": {"stat": "wisdom", "change": 1, "description": f"Gained knowledge about the {entity}"}
                })
                entity_options.append({
                    "text": f"Search for secrets within the {entity}",
                    "consequence": {"stat": "wisdom", "change": 2, "description": f"Discovered hidden aspects of the {entity}"}
                })
            elif entity in ["sword", "staff", "wand", "scroll", "potion"]:
                entity_options.append({
                    "text": f"Examine the {entity} closely",
                    "consequence": {"stat": "magic" if entity in ["wand", "scroll", "potion"] else "strength", "change": 1, "description": f"Learned more about the {entity}"}
                })
                entity_options.append({
                    "text": f"Try to use the {entity}",
                    "consequence": {"stat": "magic" if entity in ["wand", "scroll", "potion"] else "strength", "change": 2, "description": f"Practiced with the {entity}"}
                })
            elif entity in ["door", "gate", "bridge"]:
                entity_options.append({
                    "text": f"Cross the {entity} carefully",
                    "consequence": {"stat": "agility", "change": 1, "description": f"Navigated the {entity} successfully"}
                })
                entity_options.append({
                    "text": f"Inspect the {entity} for traps",
                    "consequence": {"stat": "wisdom", "change": 1, "description": f"Examined the {entity} thoroughly"}
                })
            elif entity in ["chest", "altar", "statue", "monument"]:
                entity_options.append({
                    "text": f"Inspect the {entity} carefully",
                    "consequence": {"stat": "wisdom", "change": 1, "description": f"Examined the {entity} thoroughly"}
                })
                entity_options.append({
                    "text": f"Look for hidden mechanisms on the {entity}",
                    "consequence": {"stat": "wisdom", "change": 2, "description": f"Sought secrets of the {entity}"}
                })
            elif entity in ["dragon", "creature", "beast"]:
                entity_options.append({
                    "text": f"Observe the {entity} from a safe distance",
                    "consequence": {"stat": "wisdom", "change": 1, "description": f"Learned about the {entity}'s behavior"}
                })
                entity_options.append({
                    "text": f"Try to communicate with the {entity}",
                    "consequence": {"stat": "charisma", "change": 2, "description": f"Attempted contact with the {entity}"}
                })
            elif entity in ["stranger", "wizard", "warrior"]:
                entity_options.append({
                    "text": f"Approach the {entity}",
                    "consequence": {"stat": "charisma", "change": 1, "description": f"Initiated contact with the {entity}"}
                })
                entity_options.append({
                    "text": f"Ask the {entity} for assistance",
                    "consequence": {"stat": "charisma", "change": 2, "description": f"Requested help from the {entity}"}
                })
    
    # Filter out entity options that were used previously
    filtered_entity_options = [option for option in entity_options if option["text"] not in previous_options]
    if filtered_entity_options:
        random.shuffle(filtered_entity_options)
        if len(filtered_entity_options) > 0 and len(final_options) < 3:
            final_options.append(filtered_entity_options[0])
    
    # Get options from the primary situation, filtered to remove any that match previous options
    primary_options = [option for option in all_situation_options[primary_situation] if option["text"] not in previous_options]
    secondary_options = [option for option in all_situation_options[secondary_situation] if option["text"] not in previous_options]
    
    # Combine and shuffle options
    combined_options = primary_options + secondary_options
    random.shuffle(combined_options)
    
    # Add situation-specific options to fill remaining slots
    for option in combined_options:
        if option["text"] not in [o["text"] for o in final_options] and option["text"] not in previous_options and len(final_options) < 3:
            final_options.append(option)
    
    # Fallback options with consequences
    fallback_options = [
        {"text": "Continue exploring cautiously", "consequence": {"stat": "health", "change": 1, "description": "Safety awareness increased"}},
        {"text": "Look for another path", "consequence": {"stat": "agility", "change": 1, "description": "Found an alternative route"}},
        {"text": "Rest and recover strength", "consequence": {"stat": "health", "change": 2, "description": "Recovered from fatigue"}},
        {"text": "Take a moment to observe", "consequence": {"stat": "wisdom", "change": 1, "description": "Perception improved"}},
        {"text": "Make a plan before proceeding", "consequence": {"stat": "wisdom", "change": 1, "description": "Strategy development improved"}},
        {"text": "Set up a temporary camp", "consequence": {"stat": "health", "change": 2, "description": "Rested and recovered"}},
        {"text": "Check your equipment", "consequence": {"stat": "strength", "change": 1, "description": "Equipment maintenance improved"}},
        {"text": "Consult your map", "consequence": {"stat": "wisdom", "change": 1, "description": "Navigation skills improved"}},
        {"text": "Meditate to focus your mind", "consequence": {"stat": "magic", "change": 1, "description": "Mental clarity enhanced"}}
    ]
    
    # Filter fallback options to avoid repetition
    filtered_fallbacks = [option for option in fallback_options if option["text"] not in previous_options]
    
    # If we still don't have 3 options, add some defaults
    while len(final_options) < 3 and filtered_fallbacks:
        random_option = random.choice(filtered_fallbacks)
        if random_option["text"] not in [o["text"] for o in final_options]:
            final_options.append(random_option)
            filtered_fallbacks.remove(random_option)
    
    # If we're still short on options (unusual), use unfiltered fallbacks
    while len(final_options) < 3:
        random_option = random.choice(fallback_options)
        if random_option["text"] not in [o["text"] for o in final_options]:
            final_options.append(random_option)
    
    # Sometimes add a negative consequence to create risk
    if random.random() < 0.3 and len(final_options) == 3:  # 30% chance to add a risky option
        idx = random.randint(0, 2)
        stat = random.choice(["strength", "health", "agility", "magic"])
        final_options[idx]["consequence"] = {
            "stat": stat,
            "change": -2,
            "description": f"Encountered difficulty, {stat} reduced"
        }
    
    return final_options[:3]

def generate_fallback_story(character_name, location=None, previous_choice=None):
    """Generate a simple story when AI generation is unavailable"""
    
    # Create a list of possible story snippets
    story_templates = [
        f"{character_name} ventured deeper into the {location or 'forest'}, the sounds of wildlife echoing all around. Sunlight filtered through the dense canopy above, casting dappled shadows on the forest floor. Ahead, the path split in three directions, each promising new adventures and unknown dangers.",
        
        f"As {character_name} continued the journey, the terrain became more challenging. Rocky outcroppings and thick underbrush made progress slow. In the distance, smoke could be seen rising above the trees. Something or someone awaited discovery ahead.",
        
        f"The mysterious voices grew louder as {character_name} pressed forward. Ancient ruins emerged from the mist, their weathered stones covered in strange symbols. The air hummed with magical energy, and there was a sense that important choices would need to be made soon.",
        
        f"{character_name} found a narrow path leading to a small clearing. In the center stood an ancient stone altar, its surface inscribed with runes that glowed faintly in the dim light. The atmosphere felt charged with mysterious energy. What secrets might this place hold?",
        
        f"The cave entrance loomed before {character_name}, dark and foreboding. Cool air flowed outward, carrying strange scents and distant echoes. Local legends spoke of both treasures and dangers within these depths. A decision had to be made about how to proceed."
    ]
    
    # Select a random story snippet
    return random.choice(story_templates)


REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Initialize Replicate client
replicate_client = None
if REPLICATE_API_TOKEN:
    try:
        import replicate
        replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        print("Replicate client initialized successfully")
    except Exception as e:
        print(f"Error initializing Replicate client: {e}")
        replicate_client = False
else:
    print("Warning: REPLICATE_API_TOKEN not set. Image generation with Flux will not work.")
    replicate_client = False

# Pre-fetch default version for the Flux model
REPLICATE_MODEL_NAME = "black-forest-labs/flux-schnell"
FLUX_DEFAULT_VERSION = None
if REPLICATE_API_TOKEN:
    try:
        resp = requests.get(
            f"https://api.replicate.com/v1/models/{REPLICATE_MODEL_NAME}",
            headers={"Authorization": f"Token {REPLICATE_API_TOKEN}"}
        )
        if resp.status_code == 200:
            FLUX_DEFAULT_VERSION = resp.json().get("default_version")
            print(f"Using Flux model version: {FLUX_DEFAULT_VERSION}")
        else:
            print(f"Could not fetch Flux default version, status {resp.status_code}")
    except Exception as e:
        print(f"Error retrieving default Flux version: {e}")

@app.post("/generate-image", response_model=ImageResponse)
async def generate_image(req: ImageRequest):
    """
    Generate an image based on the story text content using Replicate's Flux model.
    """
    try:
        # Extract story content and character name
        story_text = req.story_text
        character_name = req.character_name or "adventurer"
        
        # Create a detailed image prompt
        prompt = create_image_prompt(story_text, character_name)
        
        # Add a timestamp to ensure we get a unique image each time
        timestamp = int(time.time())
        
        # Try to generate with Replicate
        if REPLICATE_API_TOKEN:
            try:
                # Configure replicate client
                replicate.Client(api_token=REPLICATE_API_TOKEN)
                
                logger.info(f"Generating image with Replicate Flux model: {prompt[:100]}...")
                
                # Call the Flux model with 1:1 square aspect ratio
                output = replicate_client.run(
                    "black-forest-labs/flux-schnell",
                    input={
                        "width": 1024,  # Increased size for better quality
                        "height": 1024,  # Keep 1:1 aspect ratio
                        "prompt": prompt,
                        "num_inference_steps": 4,
                        "guidance_scale": 7.5,
                        "scheduler": "EulerA",
                        "negative_prompt": "bad anatomy, bad proportions, blurry, cropped, cross-eyed, deformed, disfigured, duplicate, error, extra arms, extra fingers, extra legs, extra limbs, fused fingers, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed limbs, missing arms, missing legs, morbid, mutated hands, mutation, mutilated, out of frame, poorly drawn face, poorly drawn hands, signature, text, too many fingers, ugly, username, watermark, worst quality"
                    }
                )
                
                if output:
                    try:
                        # Handle FileOutput type from Replicate
                        image_url = str(output) if isinstance(output, (str, replicate.helpers.FileOutput)) else output[0]
                        
                        # Convert WebP URL to string if needed
                        if hasattr(image_url, 'url'):
                            image_url = image_url.url
                        
                        if isinstance(image_url, str):
                            # Generate a unique filename, convert webp to png in filename
                            filename = f"{character_name.replace(' ', '_')}_{timestamp}.png"
                            local_path = os.path.join(OUTPUT_DIR, filename)
                            
                            # Download the image
                            response = requests.get(image_url, stream=True)
                            if response.status_code == 200:
                                with open(local_path, 'wb') as f:
                                    for chunk in response.iter_content(1024):
                                        f.write(chunk)
                                
                                # Verify the image was downloaded and is valid
                                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                                    logger.info(f"Successfully downloaded image to {local_path}")
                                    # Return the image data with success flags
                                    return ImageResponse(
                                        image_url=f"/generated_images/{filename}",
                                        prompt=prompt,
                                        success=True,
                                        generation_method="replicate_flux",
                                        image_path=local_path
                                    )
                            else:
                                logger.error(f"Failed to download image: HTTP {response.status_code}")
                        else:
                            logger.error(f"Invalid image URL format: {type(image_url)}")
                    except Exception as download_err:
                        logger.error(f"Error downloading image: {str(download_err)}")
                        traceback.print_exc()
                else:
                    logger.error("No output from Replicate API")
            except Exception as e:
                logger.error(f"Error generating image with Replicate: {str(e)}")
                traceback.print_exc()
        
        # Fallback to a placeholder image if all methods fail
        return ImageResponse(
            image_url="/static/error.png",
            prompt=prompt,
            success=False,
            generation_method="fallback",
            image_path=""
        )
    
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        traceback.print_exc()
        return ImageResponse(
            image_url="/static/error.png",
            prompt="Error generating image",
            success=False,
            generation_method="error",
            image_path=""
        )

def create_image_prompt(story_text, character_name):
    """Create a detailed image prompt based on the story text and character name"""
    # Extract key elements from the story
    scene_type = extract_scene_type(story_text)
    characters = extract_characters(story_text, character_name)
    colors = extract_colors(story_text)
    objects = extract_objects(story_text)
    
    # Format the main character with more emphasis
    main_character = f"{character_name} the adventurer"
    if len(characters) > 1:
        supporting_chars = [char for char in characters if char != character_name]
        characters_text = f"{main_character} with {', '.join(supporting_chars)}"
    else:
        characters_text = main_character
    
    # Construct a detailed scene description
    scene_desc = f"fantasy {scene_type} scene featuring {characters_text}"
    
    if colors:
        scene_desc += f", with {', '.join(colors)} color palette"
    
    if objects:
        scene_desc += f", including {', '.join(objects)}"
        
    # Create the final prompt with style elements
    prompt = (
        f"masterpiece fantasy art illustration of {scene_desc}. "
        f"highly detailed, intricate, cinematic lighting, epic composition, "
        f"dramatic scene, vibrant colors, professional fantasy concept art, "
        f"trending on ArtStation, unreal engine 5 rendering, 8k resolution"
    )
    
    return prompt

# Helper functions for image prompt creation
def extract_scene_type(text):
    """Extract the primary scene type from the story text"""
    scene_types = {
        "forest": ["forest", "woods", "jungle", "trees", "woodland"],
        "mountain": ["mountain", "peak", "cliff", "highland", "rocky"],
        "cave": ["cave", "cavern", "tunnel", "underground", "darkness"],
        "village": ["village", "town", "settlement", "buildings", "houses"],
        "castle": ["castle", "fortress", "keep", "palace", "tower"],
        "dungeon": ["dungeon", "prison", "cell", "labyrinth", "catacomb"],
        "desert": ["desert", "sand", "dunes", "wasteland", "arid"],
        "ocean": ["ocean", "sea", "beach", "coast", "waves"],
        "river": ["river", "stream", "waterfall", "lake", "creek"],
        "ruins": ["ruins", "ancient", "temple", "forgotten", "abandoned"]
    }
    
    scene_counts = {}
    for scene, keywords in scene_types.items():
        count = 0
        for keyword in keywords:
            count += len(re.findall(r'\b' + keyword + r'\b', text.lower()))
        scene_counts[scene] = count
    
    max_scene = max(scene_counts.items(), key=lambda x: x[1])
    return max_scene[0] if max_scene[1] > 0 else "fantasy"

def extract_characters(text, main_character):
    """Extract character descriptions from the story text"""
    characters = [main_character]
    potential_names = re.findall(r'\b[A-Z][a-z]+\b', text)
    
    common_words = [
        "The", "A", "An", "This", "That", "These", "Those", "It", "I", "You", "He", "She", "They",
        "If", "In", "On", "At", "By", "For", "With", "To", "From", "Of", "And", "Or", "But",
        "When", "Where", "What", "Why", "How", "Try", "Keep", "Write", "World", "Story"
    ]
    potential_names = [name for name in potential_names if name not in common_words]
    
    for name in potential_names:
        if name not in characters and len(name) > 2:
            characters.append(name)
    
    return characters[:3]

def extract_colors(text):
    """Extract color descriptions from the story text"""
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", 
              "white", "gray", "brown", "gold", "silver", "bronze", "copper"]
    
    found_colors = []
    for color in colors:
        if re.search(r'\b' + color + r'\b', text.lower()):
            found_colors.append(color)
    
    return found_colors[:3]

def extract_objects(text):
    """Extract key objects mentioned in the story text"""
    objects = ["sword", "shield", "staff", "wand", "bow", "arrow", "axe", "dagger", 
               "potion", "scroll", "book", "map", "key", "crown", "throne", "gem", 
               "crystal", "ring", "amulet", "cloak", "helm", "boots", "glove"]
    
    found_objects = []
    for obj in objects:
        if re.search(r'\b' + obj + r'\b', text.lower()):
            found_objects.append(obj)
    
    return found_objects[:3]

# Start the server when run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
