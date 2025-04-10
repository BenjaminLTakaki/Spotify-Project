import sys
print(sys.path)
import os
import json
import random
import string
from pathlib import Path  # Added for better path handling
from io import BytesIO

from flask import Flask, request, render_template, send_from_directory
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import torch
print("torch version:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("Current CUDA Device Index:", torch.cuda.current_device())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    print("Warning: CUDA is not available. Check your PyTorch installation and GPU drivers.")

from diffusers import DiffusionPipeline
from transformers import pipeline as text_pipeline
from PIL import Image
from dotenv import load_dotenv  

# Create directories for model cache and generated covers
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = BASE_DIR / "model_cache"
COVERS_DIR = BASE_DIR / "generated_covers"

# Create directories if they don't exist
CACHE_DIR.mkdir(exist_ok=True)
COVERS_DIR.mkdir(exist_ok=True)

# Set Hugging Face cache directory
os.environ["HF_HOME"] = str(CACHE_DIR)

load_dotenv()  

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise ValueError("Please set the SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in the .env file.")

print("Spotify Client ID:", SPOTIFY_CLIENT_ID)
print("Spotify Client Secret:", SPOTIFY_CLIENT_SECRET)

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    ),
    requests_timeout=15
)

# Create a torch device object (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("CUDA is available! Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Warning: CUDA not available. Using CPU.")

try:
    import accelerate
except ImportError:
    print("Warning: accelerate not installed. Install it for faster and less memory-intense model loading: pip install accelerate")

# Initialize the stable diffusion pipeline once
print("Loading Stable Diffusion XL model...")
model_name = "stabilityai/stable-diffusion-xl-base-1.0"
model_cache_dir = CACHE_DIR / model_name.replace("/", "_")
model_cache_dir.mkdir(exist_ok=True)

# Load the pipeline once
if device.type == "cuda":
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        variant="fp16",  
        use_safetensors=True,
        cache_dir=model_cache_dir  # Use model cache directory
    ).to(device)
    pipe.unet = pipe.unet.half()
    pipe.text_encoder = pipe.text_encoder.half()
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae = pipe.vae.half()
else:
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        use_safetensors=True,
        cache_dir=model_cache_dir  # Use model cache directory
    ).to(device)

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("Enabled xformers memory efficient attention")
except ModuleNotFoundError:
    print("xformers is not installed. Continuing without memory-efficient attention.")
except Exception as e:
    print(f"Could not enable xformers: {e}")

# Initialize the title generator
print("Loading title generator model...")
device_id = 0 if device.type == "cuda" else -1
title_generator = text_pipeline(
    "text-generation",
    model="EleutherAI/gpt-j-6B",
    device=device_id,
    model_kwargs={"cache_dir": CACHE_DIR / "title_generator"}  # Set cache directory for the model
)

# Assume templates already exist
TEMPLATES_DIR = BASE_DIR / "templates"
app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

def generate_random_string(size=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))

def extract_playlist_genres(playlist_url):
    """
    For each track in the playlist, retrieve the first artist's genres.
    Returns a comma-separated string of unique genres.
    """
    playlist_id = playlist_url.split("/")[-1].split("?")[0]
    results = sp.playlist_tracks(playlist_id)
    genre_set = set()
    for item in results.get("items", []):
        track = item.get("track")
        if track and track.get("artists"):
            first_artist = track["artists"][0]
            artist_info = sp.artist(first_artist["id"])
            for genre in artist_info.get("genres", []):
                genre_set.add(genre)
    if not genre_set:
        return "various genres"
    return ", ".join(genre_set)

def generate_cover_image(prompt):
    """
    Generate a cover image using the global diffusion pipeline.
    Applies a negative prompt to discourage human depictions.
    """
    # We're using the globally initialized pipeline
    global pipe
    
    print(f"Generating image with prompt: {prompt}")
    
    # Enhanced negative prompt to ensure no human figures
    negative_prompt = (
        "no humans, no faces, no body parts, no humanoid figures, no people, "
        "no portraits, no crowds, no human-like creatures, no person, no man, no woman, "
        "no child, no hands, no fingers, avoid any recognizable human features, "
        "no human subjects, no human elements, no human presence, no human forms"
    )
    
    try:
        # Use higher guidance scale to better enforce negative prompts
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=9.5  # Increased from 7.5 to better enforce negative prompts
        )
        image = result.images[0]
        
        # Clean up CUDA memory after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return image
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return Image.new('RGB', (512, 512), color='black')

def generate_title(genres, mood=""):
    """
    Generate a unique, evocative album title for a playlist using its overall mood
    and aggregated genres. The title should be imaginative and not simply repeat
    the genres.
    """
    prompt = (
        "Create a unique and evocative album title for a music playlist. "
        "The album should feel original and creative. "
    )
    if mood:
        prompt += f"It has a {mood} mood, "
    prompt += f"and is characterized by the following genres: {genres}. "
    prompt += "Do not repeat or list the genres in the title; instead, invent a new, abstract title. Title:"
    
    output = title_generator(
        prompt,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7
    )
    full_text = output[0]['generated_text']
    if "Title:" in full_text:
        title = full_text.split("Title:")[1].strip()
    else:
        title = full_text.strip()
    return title.split("\n")[0][:50]  # Limit to first 50 characters

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        playlist_url = request.form.get("playlist_url")
        mood = request.form.get("mood", "").strip()
        if not playlist_url:
            return render_template("index.html", error="Please enter a Spotify playlist URL.")
            
        print(f"Processing playlist: {playlist_url}")
        genres = extract_playlist_genres(playlist_url)
        print(f"Extracted genres: {genres}")
        
        if mood:
            image_prompt = f"An artistic album cover for a playlist with a {mood} mood, inspired by the genres: {genres}"
        else:
            image_prompt = f"An artistic album cover inspired by the genres: {genres}"
            
        cover_image = generate_cover_image(image_prompt)
        
        # Save to the covers directory with a random filename
        img_filename = generate_random_string() + ".png"
        img_path = COVERS_DIR / img_filename
        print(f"Saving image to {img_path}")
        cover_image.save(img_path)
        
        title = generate_title(genres, mood)
        print(f"Generated title: {title}")
        
        return render_template(
            "result.html",
            title=title,
            image_file=img_filename,
            playlist_keywords=genres,
            mood=mood
        )
    else:
        return render_template("index.html")

@app.route("/generated_covers/<path:filename>")
def serve_image(filename):
    return send_from_directory(COVERS_DIR, filename)

if __name__ == "__main__":
    print("Starting Flask server on port 50")
    app.run(debug=True, host="0.0.0.0", port=50)