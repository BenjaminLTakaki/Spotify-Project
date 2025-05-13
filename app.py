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

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import io
import base64

import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline as text_pipeline
from PIL import Image
from dotenv import load_dotenv\

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
    global using_fallback, sp
    
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
            temp_sp = spotipy.Spotify(auth_manager=auth_manager, requests_timeout=60, retries=3)
            test = temp_sp.search(q='test', limit=1)
            if test and 'tracks' in test:
                sp = temp_sp
            else:
                raise Exception("Test failed")
        except Exception as e:
            print(f"Error creating Spotify client: {e}")
            using_fallback = True
            return get_fallback_data("Failed to initialize Spotify client")
    
    # Parse URL and check validity
    if "playlist/" not in playlist_url and "album/" not in playlist_url:
        return {"error": "Invalid Spotify URL format"}
    
    try:
        is_playlist = "playlist/" in playlist_url
        
        try:
            if is_playlist:
                item_id = playlist_url.split("playlist/")[-1].split("?")[0].split("/")[0]
                playlist_info = sp.playlist(item_id, fields="name,description")
                item_name = playlist_info.get("name", "Unknown Playlist")
                
                # Get tracks to analyze genres
                results = sp.playlist_tracks(
                    item_id,
                    fields="items(track(id,name,artists(id,name)))",
                    market="US",
                    limit=50  
                )
                tracks = results.get("items", [])
            else: 
                item_id = playlist_url.split("album/")[-1].split("?")[0].split("/")[0]
                album_info = sp.album(item_id)
                item_name = album_info.get("name", "Unknown Album")
                
                album_tracks = album_info.get("tracks", {}).get("items", [])[:50]
                tracks = [{"track": track} for track in album_tracks]
                
            print(f"Found {'playlist' if is_playlist else 'album'}: {item_name}")
        except spotipy.exceptions.SpotifyException as e:
            print(f"Error accessing Spotify: {e}")
            using_fallback = True
            return get_fallback_data(f"Error accessing Spotify: {str(e)}", "Unknown Item")
        
        if not tracks:
            using_fallback = True
            return get_fallback_data("No tracks found", item_name)
        
        # Extract all artist IDs from tracks
        artists = []
        track_names = []
        
        for item in tracks:
            track = item.get("track")
            if track and track.get("name"):
                track_names.append(track.get("name"))
                
            if track and track.get("artists"):
                for artist in track.get("artists"):
                    if artist.get("id"):
                        artists.append(artist.get("id"))
        
        if not artists:
            using_fallback = True
            return get_fallback_data("No artists found in tracks", item_name)
            
        print(f"Found {len(set(artists))} unique artists in {len(tracks)} tracks")
        
        # Get genres from artists
        genres = []
        unique_artist_ids = list(set(artists))[:50]  # Limit to 50 artists max for API calls
        
        try:
            # Process artists in batches (Spotify API allows up to 50 per request)
            for i in range(0, len(unique_artist_ids), 50):
                batch = unique_artist_ids[i:min(i+50, len(unique_artist_ids))]
                
                try:
                    artist_info_batch = sp.artists(batch)
                    if artist_info_batch and 'artists' in artist_info_batch:
                        for artist in artist_info_batch['artists']:
                            artist_genres = artist.get('genres', [])
                            genres.extend(artist_genres)
                except Exception as e:
                    print(f"Error fetching artist genres: {e}")
                    continue
            
            # Count and sort genres by frequency
            genre_counter = Counter(genres)
            top_genres = [genre for genre, _ in genre_counter.most_common(10)]
            genres_with_counts = genre_counter.most_common(20)  # Keep more for visualization
            
            if not top_genres:
                using_fallback = True
                top_genres = ["electronic", "pop", "indie", "rock", "alternative"]
                genres_with_counts = [("electronic", 3), ("pop", 2), ("indie", 2), ("rock", 1), ("alternative", 1)]
            
            print(f"Top genres: {', '.join(top_genres[:5])}")
            
        except Exception as e:
            print(f"Error processing genres: {e}")
            using_fallback = True
            top_genres = ["electronic", "pop", "indie", "rock", "alternative"]
            genres_with_counts = [("electronic", 3), ("pop", 2), ("indie", 2), ("rock", 1), ("alternative", 1)]
        
        # Determine overall mood based on genres
        mood = "balanced"  # Default mood
        
        # Simple genre-based mood classification
        mood_keywords = {
            "euphoric": ["edm", "dance", "house", "electronic", "pop", "party"],
            "energetic": ["rock", "metal", "punk", "trap", "dubstep"],
            "peaceful": ["ambient", "classical", "chill", "lo-fi", "instrumental"],
            "melancholic": ["sad", "slow", "ballad", "emotional", "soul", "blues"],
            "upbeat": ["happy", "funk", "disco", "pop", "tropical"],
            "relaxed": ["acoustic", "folk", "indie", "soft", "ambient"]
        }
        
        # Count genre matches for each mood
        mood_scores = {mood: 0 for mood in mood_keywords}
        
        for genre in genres:
            for mood_name, keywords in mood_keywords.items():
                if any(keyword in genre.lower() for keyword in keywords):
                    mood_scores[mood_name] += 1
        
        # Pick highest scoring mood if we have matches
        if any(mood_scores.values()):
            mood = max(mood_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate energy level based on genres
        low_energy_genres = ["ambient", "classical", "chill", "lo-fi", "acoustic", "folk"]
        high_energy_genres = ["rock", "metal", "edm", "dance", "trap", "dubstep", "house"]
        
        low_count = sum(1 for genre in genres if any(keyword in genre.lower() for keyword in low_energy_genres))
        high_count = sum(1 for genre in genres if any(keyword in genre.lower() for keyword in high_energy_genres))
        
        if high_count > low_count:
            energy_level = "energetic"
        elif low_count > high_count:
            energy_level = "calm"
        else:
            energy_level = "balanced"
        
        # Return complete data
        return {
            "track_names": track_names[:10],
            "genres": top_genres[:10],
            "all_genres": genres,  # Keep all genres for visualization
            "genres_with_counts": genres_with_counts,  # For chart generation
            "energy_level": energy_level,
            "mood_descriptor": mood,
            "item_name": item_name,
            "using_fallback": using_fallback,
            "found_genres": bool(genres)
        }
        
    except Exception as e:
        print(f"Error extracting data: {e}")
        using_fallback = True
        return get_fallback_data(f"Error: {str(e)}", "Unknown Item")

def generate_genre_chart(genres):
    """Generate a bar chart visualization of genres"""
    if not genres:
        genres = ["electronic", "pop", "indie"]
    
    # Count genre frequencies if it's a list
    if isinstance(genres, list):
        # Count genres by frequency using Counter
        genre_counter = Counter(genres)
        genres_sorted = genre_counter.most_common(8)  # Get top 8 genres
        labels = [genre for genre, _ in genres_sorted]
        values = [count for _, count in genres_sorted]
    else:
        # Fallback if genres is not a list
        labels = ["electronic", "pop", "indie", "rock", "alternative"]
        values = [3, 2, 2, 1, 1]
    
    # Create bar chart with improved styling
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Plot bars with Spotify green color gradient
    bars = ax.bar(
        labels, values, 
        color=['#1DB954', '#1ED760', '#24E066', '#30D67B', '#3DCF8B', '#4AC89D', '#57C0AD', '#63B9BE'][:len(labels)],
        width=0.6, edgecolor='#444444'
    )
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{int(height)}', ha='center', va='bottom', 
            color='white', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='#333', alpha=0.7)
        )
    
    # Set labels and title with improved font
    ax.set_xlabel('Genres', fontsize=14, fontweight='bold', color='white', labelpad=10)
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold', color='white', labelpad=10)
    ax.set_title('Genre Analysis', fontsize=18, fontweight='bold', color='#1DB954', pad=20)
    
    # Configure chart aesthetics
    ax.set_facecolor('#2a2a2a')
    fig.patch.set_facecolor('#1e1e1e')
    
    # Style the axis and ticks - MODIFIED LINE BELOW
    ax.tick_params(axis='x', colors='white', labelsize=12, rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")  # Set horizontal alignment separately
    
    ax.tick_params(axis='y', colors='white', labelsize=12)
    
    # Add subtle grid lines
    ax.grid(axis='y', color='#444444', alpha=0.3, linestyle='--')
    
    # Adjust layout for better fit with rotated labels
    plt.tight_layout()
    
    # Save to base64 for embedding in HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', transparent=True, dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)  # Close figure to free memory
    
    return f"data:image/png;base64,{image_base64}"

def get_fallback_data(reason, item_name="Playlist"):
    """Create fallback data with a reason"""
    print(f"Using fallback data: {reason}")
    fallback_genres = ["electronic", "pop", "indie", "rock", "alternative"]
    fallback_genre_counts = [("electronic", 3), ("pop", 2), ("indie", 2), ("rock", 1), ("alternative", 1)]
    
    return {
        "track_names": ["Unknown Track"],
        "genres": fallback_genres,
        "all_genres": fallback_genres,
        "genres_with_counts": fallback_genre_counts,
        "energy_level": "balanced",
        "mood_descriptor": "upbeat",
        "item_name": item_name,
        "using_fallback": True,
        "fallback_reason": reason,
        "found_genres": False
    }

def create_prompt_from_data(playlist_data, user_mood=None):
    """Create optimized prompt for stable diffusion"""
    genres_str = ", ".join(playlist_data.get("genres", ["various"]))
    energy = playlist_data.get("energy_level", "balanced")
    mood = user_mood if user_mood else playlist_data.get("mood_descriptor", "balanced")
    
    prompt = (
        f"album cover art, {genres_str} music, professional artwork, "
        f"highly detailed, 8k"
    )
    
    if user_mood:
        prompt += f", {user_mood} atmosphere"
    
    genres_lower = [g.lower() for g in playlist_data.get("genres", [])]
    
    style_elements = []
    # Feature extraction for better prompt based on genres
    if any("rock" in g for g in genres_lower) or any("metal" in g for g in genres_lower):
        style_elements.append("dramatic lighting, bold contrasts")
    elif any("electronic" in g for g in genres_lower) or any("techno" in g for g in genres_lower):
        style_elements.append("futuristic, digital elements, abstract patterns")
    elif any("hip hop" in g for g in genres_lower) or any("rap" in g for g in genres_lower):
        style_elements.append("urban aesthetic, stylish, street art influence")
    elif any("jazz" in g for g in genres_lower) or any("blues" in g for g in genres_lower):
        style_elements.append("smoky atmosphere, classic vibe, vintage feel")
    elif any("folk" in g for g in genres_lower) or any("acoustic" in g for g in genres_lower):
        style_elements.append("organic textures, natural elements, warm tones")
    
    if style_elements:
        prompt += ", " + ", ".join(style_elements)
    
    return prompt

def generate_cover_image(prompt):
    if not pipe:
        return Image.new('RGB', (512, 512), color='#3A506B')
    
    print(f"Generating with prompt: {prompt}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        negative_prompt = "text, words, letters, signature, watermark, low quality, blurry, pixelated, human, person, people, man, woman, child, baby, face, portrait, body, skin, hands, fingers, arms, legs, feet, eyes, nose, mouth, ears, hair, facial features, silhouette, human figure, human form, character, human-like, anthropomorphic, mannequin, crowd, group, person standing, walking figure, sitting person, human anatomy, bad anatomy, bad proportions, distorted anatomy, extra limbs, missing limbs, fused limbs, poorly drawn face, cloned face, asymmetrical face, multiple faces, strange anatomy, deformed body, mutation, mutilated, disfigured, malformed limbs, extra fingers, missing fingers, too many fingers, elongated limbs, human skin texture, human clothing, naked figure, nude, facial expressions, human poses, humanoid, human-shaped, human outline, human shadow, human contours, human dimensions, human likeness, human presence, human attributes, human characteristics, human appearance, human form, skeleton, skull, rib cage, spine, realistic skin, realistic body, realistic person"
        
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,  
            guidance_scale=13,
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
        try:
            playlist_url = request.form.get("playlist_url")
            user_mood = request.form.get("mood", "").strip()
            
            if not playlist_url:
                return render_template("index.html", error="Please enter a Spotify playlist or album URL.")
            
            # Process the playlist or album
            playlist_data = extract_playlist_data(playlist_url)
            
            # Handle URL format errors
            if "error" in playlist_data and "Invalid Spotify URL format" in playlist_data["error"]:
                return render_template("index.html", error="Invalid Spotify URL format. Please enter a valid Spotify playlist or album URL.")
            
            # Generate cover image
            image_prompt = create_prompt_from_data(playlist_data, user_mood)
            cover_image = generate_cover_image(image_prompt)
            
            # Save image
            img_filename = generate_random_string() + ".png"
            cover_image.save(COVERS_DIR / img_filename)
            
            # Generate title
            title = generate_title(playlist_data, user_mood)
            
            # Generate genre chart
            genres_chart = generate_genre_chart(playlist_data.get("all_genres", []))
            
            # Data for display
            display_data = {
                "title": title,
                "image_file": img_filename,
                "genres": ", ".join(playlist_data.get("genres", [])),
                "mood": user_mood if user_mood else playlist_data.get("mood_descriptor", ""),
                "energy": playlist_data.get("energy_level", ""),
                "playlist_name": playlist_data.get("item_name", "Your Music"),
                "using_fallback": using_fallback,
                "found_genres": playlist_data.get("found_genres", False),
                "genres_chart": genres_chart
            }
            
            return render_template("result.html", **display_data)
        except Exception as e:
            print(f"Server error processing request: {e}")
            return render_template("index.html", error=f"An error occurred: {str(e)}")
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