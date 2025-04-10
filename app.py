import os
import json
import random
import string
import statistics
from pathlib import Path
from collections import Counter

from flask import Flask, request, render_template, send_from_directory, jsonify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline as text_pipeline
from PIL import Image
from dotenv import load_dotenv

#Setup paths and environment 
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = BASE_DIR / "model_cache"
COVERS_DIR = BASE_DIR / "generated_covers"
CACHE_DIR.mkdir(exist_ok=True)
COVERS_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

#Load environment variables 
load_dotenv()  
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

#Initialize global variables 
sp = None
pipe = None
title_generator = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_initialized = False
using_fallback = False

#Initialize Flask app
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

#Helper functions
def generate_random_string(size=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))

def initialize_models():
    """Initialize all models once at startup"""
    global sp, pipe, title_generator, device, model_initialized
    
    print(f"Device: {device} - RTX 4080 Super (16GB VRAM)")
    
    # Initialize Spotify client
    print("Initializing Spotify client...")
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            cache_handler=None
        )
        
        sp = spotipy.Spotify(
            auth_manager=auth_manager,
            requests_timeout=30,
            retries=3
        )
        
        # Quick test
        sp.search(q='test', limit=1)
        print("✓ Spotify API connection successful")
    except Exception as e:
        print(f"✗ Spotify API initialization failed: {e}")
        sp = None
    
    try:
        print("Loading Stable Diffusion model...")
        # Using SD v1.5 for better performance on 16GB VRAM
        model_id = "runwayml/stable-diffusion-v1-5"
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use half precision for 16GB VRAM
            cache_dir=CACHE_DIR / model_id.replace("/", "_")
        ).to(device)
        
        pipe.enable_attention_slicing()
        pipe.enable_xformers_memory_efficient_attention()
        torch.backends.cudnn.benchmark = True
        
        print("✓ Image generation model loaded")
    except Exception as e:
        print(f"✗ Error loading image model: {e}")
        pipe = None
    
    # Load smaller title generator
    try:
        print("Loading title generator...")
        title_generator = text_pipeline(
            "text-generation",
            model="distilgpt2",
            device=0 if device.type == "cuda" else -1
        )
        print("✓ Title generator loaded")
    except Exception as e:
        print(f"✗ Error loading title generator: {e}")
        title_generator = None
    
    model_initialized = True
    return all([sp, pipe, title_generator])

def extract_playlist_data(playlist_url):
    """Extract data from playlist with fallback indicators"""
    global using_fallback
    
    # Reset fallback indicator
    using_fallback = False
    
    # Check Spotify client
    if not sp:
        print("Attempting to create new Spotify client...")
        try:
            auth_manager = SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET
            )
            temp_sp = spotipy.Spotify(auth_manager=auth_manager)
            test = temp_sp.search(q='test', limit=1)
            if test and 'tracks' in test:
                sp_client = temp_sp
            else:
                raise Exception("Test failed")
        except:
            using_fallback = True
            return get_fallback_data("Failed to initialize Spotify client")
    else:
        sp_client = sp
    
    # Get playlist ID
    if "playlist/" not in playlist_url:
        return {"error": "Invalid Spotify playlist URL format"}
    
    playlist_id = playlist_url.split("playlist/")[-1].split("?")[0].split("/")[0]
    print(f"Processing playlist ID: {playlist_id}")
    
    try:
        # Get basic playlist info
        try:
            playlist_info = sp_client.playlist(playlist_id, fields="name,description")
            playlist_name = playlist_info.get("name", "Unknown Playlist")
            print(f"Found playlist: {playlist_name}")
        except Exception as e:
            print(f"Error getting playlist info: {e}")
            playlist_name = "Unknown Playlist"
            using_fallback = True
        
        # Get tracks
        try:
            results = sp_client.playlist_tracks(
                playlist_id,
                fields="items(track(id,name,artists(id,name))),next",
                market="US",
                limit=20
            )
            tracks = results.get("items", [])
            
            if not tracks:
                using_fallback = True
                return get_fallback_data("No tracks found", playlist_name)
                
            # Extract track IDs, names, artists
            track_ids = []
            artists = []
            track_names = []
            
            for item in tracks:
                track = item.get("track")
                if track and track.get("id"):
                    track_ids.append(track.get("id"))
                    track_names.append(track.get("name"))
                    if track.get("artists"):
                        for artist in track.get("artists"):
                            if artist.get("id"):
                                artists.append(artist.get("id"))
            
            if not track_ids:
                using_fallback = True
                return get_fallback_data("No valid tracks found", playlist_name)
        except Exception as e:
            print(f"Error getting tracks: {e}")
            using_fallback = True
            return get_fallback_data("Error getting tracks", playlist_name)
        
        # Initialize default values
        audio_features_found = False
        genres_found = False
        
        # Get audio features and genres
        avg_features = {}
        genres = []
        
        # Try to get audio features
        try:
            # Test batch with fewer tracks first
            test_batch = track_ids[:3]
            features = sp_client.audio_features(test_batch)
            
            if features and any(features):
                audio_features = []
                # Get all features in smaller batches
                for i in range(0, len(track_ids), 5):
                    batch = track_ids[i:i+5]
                    batch_features = sp_client.audio_features(batch)
                    if batch_features:
                        audio_features.extend([f for f in batch_features if f])
                
                # Calculate averages
                if audio_features:
                    audio_features_found = True
                    feature_keys = [
                        "danceability", "energy", "loudness", "speechiness", 
                        "acousticness", "instrumentalness", "liveness", "valence", "tempo"
                    ]
                    
                    for key in feature_keys:
                        values = [float(af[key]) for af in audio_features if af and key in af and af[key] is not None]
                        avg_features[key] = statistics.mean(values) if values else 0.5
        except Exception as e:
            print(f"Error getting audio features: {e}")
            audio_features_found = False
        
        # Try to get genres
        try:
            # Only get 3 artists at most
            unique_artists = list(set(artists))[:3]
            
            if unique_artists:
                artists_data = sp_client.artists(unique_artists)
                if artists_data and "artists" in artists_data:
                    for artist in artists_data["artists"]:
                        genres.extend(artist.get("genres", []))
                    
                    # Remove duplicates
                    genres = list(set(genres))
                    genres_found = bool(genres)
        except Exception as e:
            print(f"Error getting genres: {e}")
            genres_found = False
        
        # Use fallbacks if needed
        if not audio_features_found:
            using_fallback = True
            print("Using fallback audio features")
            avg_features = {
                "danceability": 0.65, "energy": 0.7, "loudness": -7,
                "speechiness": 0.05, "acousticness": 0.3, "instrumentalness": 0.1,
                "liveness": 0.15, "valence": 0.6, "tempo": 120
            }
        
        if not genres_found:
            using_fallback = True
            print("Using fallback genres")
            genres = ["electronic", "pop", "indie"]
        
        # Calculate descriptors
        tempo = avg_features.get("tempo", 120)
        energy = avg_features.get("energy", 0.7)
        valence = avg_features.get("valence", 0.6)
        
        tempo_range = "slow" if tempo < 80 else "fast" if tempo >= 120 else "moderate"
        energy_level = "calm" if energy < 0.33 else "energetic" if energy >= 0.66 else "balanced"
        
        # Calculate mood
        if valence > 0.7 and energy > 0.7:
            mood = "euphoric"
        elif valence > 0.7 and energy < 0.3:
            mood = "peaceful"
        elif valence < 0.3 and energy > 0.7:
            mood = "angry"
        elif valence < 0.3 and energy < 0.3:
            mood = "melancholic"
        elif valence > 0.5 and energy > 0.5:
            mood = "upbeat"
        elif valence > 0.5 and energy < 0.5:
            mood = "relaxed"
        elif valence < 0.5 and energy > 0.5:
            mood = "tense"
        elif valence < 0.5 and energy < 0.5:
            mood = "somber"
        else:
            mood = "balanced"
        
        # Return complete data
        return {
            "track_names": track_names[:10],
            "genres": genres[:5],
            "audio_features": avg_features,
            "tempo_range": tempo_range,
            "energy_level": energy_level,
            "mood_descriptor": mood,
            "playlist_name": playlist_name,
            "using_fallback": using_fallback,
            "found_audio_features": audio_features_found,
            "found_genres": genres_found
        }
        
    except Exception as e:
        print(f"Error extracting playlist data: {e}")
        using_fallback = True
        return get_fallback_data(f"Error: {str(e)}", "Unknown Playlist")

def get_fallback_data(reason, playlist_name="Playlist"):
    """Create fallback data with a reason"""
    print(f"Using fallback data: {reason}")
    return {
        "track_names": ["Unknown Track"],
        "genres": ["electronic", "pop", "indie"],
        "audio_features": {
            "danceability": 0.65, "energy": 0.7, "loudness": -7,
            "speechiness": 0.05, "acousticness": 0.3, "instrumentalness": 0.1,
            "liveness": 0.15, "valence": 0.6, "tempo": 120
        },
        "tempo_range": "moderate",
        "energy_level": "balanced",
        "mood_descriptor": "upbeat",
        "playlist_name": playlist_name,
        "using_fallback": True,
        "fallback_reason": reason,
        "found_audio_features": False,
        "found_genres": False
    }

def create_prompt_from_data(playlist_data, user_mood=None):
    """Create optimized prompt for stable diffusion"""
    genres_str = ", ".join(playlist_data.get("genres", ["various"]))
    tempo = playlist_data.get("tempo_range", "moderate")
    energy = playlist_data.get("energy_level", "balanced")
    mood = user_mood if user_mood else playlist_data.get("mood_descriptor", "balanced")
    
    prompt = (
        f"premium album cover art, {genres_str} music, {mood} atmosphere, "
        f"{tempo} rhythm, {energy} feeling, professional artwork, "
        f"highly detailed, 8k, trending on artstation"
    )
    
    if mood == "euphoric":
        prompt += ", vibrant colors, dynamic composition"
    elif mood == "peaceful":
        prompt += ", serene landscape, soft lighting"
    elif mood == "angry":
        prompt += ", dark tones, sharp angles"
    elif mood == "melancholic":
        prompt += ", muted colors, misty atmosphere"
    
    return prompt

def generate_cover_image(prompt):
    if not pipe:
        return Image.new('RGB', (512, 512), color='#3A506B')
    
    print(f"Generating with prompt: {prompt}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Fixed and improved negative prompt
        negative_prompt = "text, words, letters, signature, watermark, low quality, blurry, pixelated"
        
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,  
            guidance_scale=7.5,
            width=768,
            height=768,
        )
        
        return result.images[0]
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return Image.new('RGB', (512, 512), color='#3A506B')

def generate_title(playlist_data, mood=""):
    """Generate album title"""
    if not title_generator:
        return "Cosmic Waves"  # Simple fallback
    
    genres = ", ".join(playlist_data.get("genres", ["music"]))
    mood_to_use = mood if mood else playlist_data.get("mood_descriptor", "balanced")
    
    prompt = f"Create a unique album title for {mood_to_use} {genres} music: "
    
    try:
        output = title_generator(
            prompt,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.8
        )
        
        # Extract just the title
        title = output[0]['generated_text'].replace(prompt, "").strip()
        return title.split("\n")[0][:40]
    except:
        # Fallback titles
        adjectives = ["Cosmic", "Velvet", "Electric", "Crystal", "Neon", "Midnight"]
        nouns = ["Waves", "Dreams", "Echo", "Horizon", "Pulse", "Journey"]
        return f"{random.choice(adjectives)} {random.choice(nouns)}"

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    global using_fallback
    
    if not model_initialized:
        if initialize_models():
            print("Models initialized successfully")
        else:
            print("Failed to initialize models")
    
    if request.method == "POST":
        playlist_url = request.form.get("playlist_url")
        user_mood = request.form.get("mood", "").strip()
        
        if not playlist_url:
            return render_template("index.html", error="Please enter a Spotify playlist URL.")
        
        # Process the playlist
        playlist_data = extract_playlist_data(playlist_url)
        
        # Handle URL format errors
        if "error" in playlist_data and "Invalid Spotify playlist URL format" in playlist_data["error"]:
            return render_template("index.html", error="Invalid Spotify playlist URL format")
        
        # Generate cover image
        image_prompt = create_prompt_from_data(playlist_data, user_mood)
        cover_image = generate_cover_image(image_prompt)
        
        # Save image
        img_filename = generate_random_string() + ".png"
        cover_image.save(COVERS_DIR / img_filename)
        
        # Generate title
        title = generate_title(playlist_data, user_mood)
        
        # Data for display
        display_data = {
            "title": title,
            "image_file": img_filename,
            "genres": ", ".join(playlist_data.get("genres", [])),
            "mood": user_mood if user_mood else playlist_data.get("mood_descriptor", ""),
            "tempo": playlist_data.get("tempo_range", ""),
            "energy": playlist_data.get("energy_level", ""),
            "features": playlist_data.get("audio_features", {}),
            "playlist_name": playlist_data.get("playlist_name", "Your Playlist"),
            "using_fallback": using_fallback,
            "found_audio_features": playlist_data.get("found_audio_features", False),
            "found_genres": playlist_data.get("found_genres", False)
        }
        
        return render_template("result.html", **display_data)
    else:
        return render_template("index.html")

@app.route("/generated_covers/<path:filename>")
def serve_image(filename):
    return send_from_directory(COVERS_DIR, filename)

@app.route("/status")
def status():
    """API endpoint to check system status"""
    return jsonify({
        "models_loaded": model_initialized,
        "spotify_working": sp is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })

if __name__ == "__main__":
    initialize_models()
    app.run(debug=False, host="0.0.0.0", port=50)