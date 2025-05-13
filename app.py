import os
import json
import random
import string
import statistics
import datetime
import io
import base64
from pathlib import Path
from collections import Counter

from flask import Flask, request, render_template, send_from_directory, jsonify, session
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

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
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

#Initialize global variables 
sp = None
pipe = None
title_generator = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_initialized = False

#Helper functions
def generate_random_string(size=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))

#Initialize Flask app
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.secret_key = os.getenv("FLASK_SECRET_KEY", generate_random_string(24))  # Add secret key for session

def initialize_spotify(use_oauth=False):
    """Initialize Spotify API client"""
    global sp
    try:
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            print("ERROR: Missing Spotify API credentials. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your .env file.")
            return False
            
        if use_oauth:
            auth_manager = SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope="user-library-read playlist-read-private playlist-read-collaborative user-read-private",
                cache_path=".spotify_cache"
            )
        else:
            auth_manager = SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET
            )
            
        sp = spotipy.Spotify(auth_manager=auth_manager, requests_timeout=60, retries=3)
        
        # Test the connection
        try:
            if use_oauth:
                sp.current_user()
            else:
                sp.search(q='test', limit=1)
            print("✓ Spotify API connection successful")
            return True
        except spotipy.exceptions.SpotifyException as e:
            print(f"✗ Spotify API authentication failed: {e}")
            # If client credentials failed, try OAuth as fallback
            if not use_oauth:
                print("Trying OAuth authentication instead...")
                return initialize_spotify(use_oauth=True)
            return False
    except Exception as e:
        print(f"✗ Spotify API initialization failed: {e}")
        return False

def initialize_image_model():
    """Initialize Stable Diffusion model"""
    global pipe
    try:
        print("Loading Stable Diffusion model...")
        model_id = "runwayml/stable-diffusion-v1-5"
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            cache_dir=CACHE_DIR / model_id.replace("/", "_")
        ).to(device)
        
        if device.type == "cuda":
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()
            if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
                pipe.enable_xformers_memory_efficient_attention()
            torch.backends.cudnn.benchmark = True
        
        print("✓ Image generation model loaded")
        return True
    except Exception as e:
        print(f"✗ Error loading image model: {e}")
        return False

def initialize_title_model():
    """Initialize Meta-Llama 3 for title generation"""
    global title_generator
    try:
        print("Loading Meta-Llama-3 title generator...")
        title_generator = text_pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            token=os.getenv("HF_TOKEN"),
            model_kwargs={"load_in_8bit": True}
        )
        print("✓ Meta-Llama-3 title generator loaded")
        return True
    except Exception as e:
        print(f"✗ Error loading Llama 3 model: {e}")
        return False

def initialize_models():
    """Initialize all models once at startup"""
    global model_initialized
    
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    spotify_initialized = initialize_spotify()
    image_model_initialized = initialize_image_model()
    title_model_initialized = initialize_title_model()
    
    model_initialized = all([spotify_initialized, image_model_initialized, title_model_initialized])
    return model_initialized

def extract_playlist_data(playlist_url):
    """Extract data from playlist"""
    global sp
    
    # Check Spotify client
    if not sp:
        print("Attempting to create new Spotify client...")
        if not initialize_spotify():
            return {"error": "Failed to initialize Spotify client"}
    
    # Parse URL and check validity
    if "playlist/" not in playlist_url and "album/" not in playlist_url:
        return {"error": "Invalid Spotify URL format"}
    
    try:
        is_playlist = "playlist/" in playlist_url
        
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
        
        if not tracks:
            return {"error": "No tracks found in the playlist or album"}
        
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
            return {"error": "No artists found in tracks"}
            
        print(f"Found {len(set(artists))} unique artists in {len(tracks)} tracks")
        
        # Get genres from artists
        genres = []
        unique_artist_ids = list(set(artists))[:50]  # Limit to 50 artists max for API calls
        
        # Process artists in batches (Spotify API allows up to 50 per request)
        for i in range(0, len(unique_artist_ids), 50):
            batch = unique_artist_ids[i:min(i+50, len(unique_artist_ids))]
            
            artist_info_batch = sp.artists(batch)
            if artist_info_batch and 'artists' in artist_info_batch:
                for artist in artist_info_batch['artists']:
                    artist_genres = artist.get('genres', [])
                    genres.extend(artist_genres)
        
        # Count and sort genres by frequency
        genre_counter = Counter(genres)
        top_genres = [genre for genre, _ in genre_counter.most_common(10)]
        genres_with_counts = genre_counter.most_common(20)  # Keep more for visualization
        
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
            "found_genres": bool(genres)
        }
        
    except Exception as e:
        print(f"Error extracting data: {e}")
        return {"error": f"Error extracting data: {str(e)}"}

def generate_genre_chart(genres):
    """Generate a bar chart visualization of genres"""
    if not genres:
        return None
    
    # Count genre frequencies if it's a list
    if isinstance(genres, list):
        # Count genres by frequency using Counter
        genre_counter = Counter(genres)
        genres_sorted = genre_counter.most_common(8)  # Get top 8 genres
        labels = [genre for genre, _ in genres_sorted]
        values = [count for _, count in genres_sorted]
    else:
        return None
    
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
    
    # Style the axis and ticks
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

def create_prompt_from_data(playlist_data, user_mood=None):
    """Create optimized prompt for stable diffusion"""
    genres_str = ", ".join(playlist_data.get("genres", ["various"]))
    energy = playlist_data.get("energy_level", "balanced")
    mood = user_mood if user_mood else playlist_data.get("mood_descriptor", "balanced")
    
    genres_lower = [g.lower() for g in playlist_data.get("genres", [])]
    
    style_elements = []
    
    prompt = (
        f"album cover art, {genres_str} music, professional artwork, "
        f"highly detailed, 8k"
    )
    
    if user_mood:
        prompt += f", {user_mood} atmosphere"
    
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
    
    # Add style elements to the playlist data for title generation
    playlist_data["style_elements"] = style_elements
    
    return prompt

def generate_cover_image(prompt, output_path=None):
    """Generate album cover image using Stable Diffusion"""
    if not pipe:
        if not initialize_image_model():
            return False
    
    print(f"Generating with prompt: {prompt}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        negative_prompt = "human, person, people, man, woman, child, baby, face, portrait, body, skin, hands, fingers, arms, legs, feet, eyes, nose, mouth, ears, hair, facial features, silhouette, human figure, human form, character, human-like, anthropomorphic, mannequin, crowd, group, person standing, walking figure, sitting person, human anatomy, bad anatomy, bad proportions, distorted anatomy, extra limbs, missing limbs, fused limbs, poorly drawn face, cloned face, asymmetrical face, multiple faces, strange anatomy, deformed body, mutation, mutilated, disfigured, malformed limbs, extra fingers, missing fingers, too many fingers, elongated limbs, human skin texture, human clothing, naked figure, nude, facial expressions, human poses, humanoid, human-shaped, human outline, human shadow, human contours, human dimensions, human likeness, human presence, human attributes, human characteristics, human appearance, human form, skeleton, skull, rib cage, spine, realistic skin, realistic body, realistic person"
        
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,  
            guidance_scale=13.5,
            width=768,
            height=768,
        )
        
        image = result.images[0]
        
        # Save image if output path is specified
        if output_path:
            image.save(output_path)
            return True
        
        return image
        
    except Exception as e:
        print(f"Error generating image: {e}")
        if output_path:
            return False
        return Image.new('RGB', (512, 512), color='#3A506B')

def generate_title(playlist_data, mood=""):
    """Generate album title using Meta-Llama 3 model"""
    if not title_generator and not initialize_title_model():
        return "New Album"
    
    genres = ", ".join(playlist_data.get("genres", ["music"]))
    mood_to_use = mood if mood else ""
    
    style_elements = playlist_data.get("style_elements", [])
    style_text = ", ".join(style_elements) if style_elements else ""
    
    prompt = f"""<|begin_of_text|><|system|>
You are an expert in creating unique, evocative album titles. Create a single, short album title (3-5 words) for a {mood_to_use} {genres} music album. 
The album cover has these visual elements: {style_text}.
Create a title that reflects both the music genres and these visual elements.
Respond with only the title, nothing else.
<|user|>
Create an album title for a {mood_to_use} {genres} album with visual elements: {style_text}
<|assistant|>
"""
    
    try:
        output = title_generator(
            prompt,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.85,
        )
        
        title = output[0]['generated_text'].replace(prompt, "").strip()
        title = title.split("\n")[0].strip().replace('"', '').replace("'", "")
        title = title.replace("<|end_of_text|>", "")
        
        return title[:50] if title and len(title) >= 3 else "New Album"
    except Exception as e:
        print(f"Error generating title: {e}")
        return "New Album"

def generate_cover(url, user_mood=None, output_path=None):
    """Generate album cover and title from Spotify URL and save data"""
    print(f"Processing Spotify URL: {url}")
    
    # Create data directory if it doesn't exist
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)
    
    # Extract playlist/album data
    data = extract_playlist_data(url)
    if "error" in data:
        return {"error": data["error"]}
    
    print(f"\nSuccessfully extracted data for: {data.get('item_name')}")
    print(f"Top genres identified: {', '.join(data.get('genres', ['unknown']))}")
    
    # Create image prompt
    base_image_prompt = create_prompt_from_data(data, user_mood)
    
    # Generate title
    title = generate_title(data, user_mood)
    print(f"Generated title: {title}")
    
    # Add title to data
    data["title"] = title
    
    # Create final image prompt with title
    image_prompt = f"{base_image_prompt}, representing the album '{title}'"
    print(f"Final image prompt with title: {image_prompt}")
    
    # Determine output path
    if not output_path:
        safe_title = "".join(c for c in title if c.isalnum() or c in [' ', '-', '_']).strip()
        safe_title = safe_title.replace(' ', '_')
        img_filename = f"{safe_title}.png"
        output_path = COVERS_DIR / img_filename
    
    # Generate cover image
    success = generate_cover_image(image_prompt, output_path)
    
    # Result info
    result = {
        "title": title,
        "output_path": str(output_path),
        "item_name": data.get("item_name"),
        "genres": data.get("genres"),
        "all_genres": data.get("all_genres"),
        "style_elements": data.get("style_elements", []),
        "mood": user_mood if user_mood else data.get("mood_descriptor", ""),
        "energy_level": data.get("energy_level", ""),
        "timestamp": str(datetime.datetime.now()),
        "spotify_url": url
    }
    
    # Save data to JSON file
    try:
        # Create a unique filename based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in data.get("item_name", "") if c.isalnum() or c in [' ', '-', '_']).strip()
        safe_name = safe_name.replace(' ', '_')
        json_filename = f"{timestamp}_{safe_name}.json"
        
        with open(DATA_DIR / json_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to {DATA_DIR / json_filename}")
        
        # Also return the filename for the saved data
        result["data_file"] = str(DATA_DIR / json_filename)
    except Exception as e:
        print(f"Error saving data: {e}")
    
    return result

def calculate_genre_percentages(genres_list):
    """Calculate percentage distribution of genres"""
    if not genres_list:
        return []
        
    # Count genres
    genre_counter = Counter(genres_list)
    total_count = sum(genre_counter.values())
    
    # Sort and get percentages
    sorted_genres = genre_counter.most_common(5)  # Get top 5 genres
    
    # Calculate percentages
    genre_percentages = [
        {"name": genre, "percentage": round((count / total_count) * 100)} 
        for genre, count in sorted_genres
    ]
    
    return genre_percentages

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
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
            
            # Use the new generate_cover function that handles the entire process
            result = generate_cover(playlist_url, user_mood)
            
            # Handle errors
            if "error" in result:
                return render_template("index.html", error=result["error"])
              # Get the filename part from the full path
            img_filename = os.path.basename(result["output_path"])
            
            # Generate genre chart
            genres_chart = generate_genre_chart(result.get("all_genres", []))
            
            # Calculate genre percentages for visualization
            genre_percentages = calculate_genre_percentages(result.get("all_genres", []))
            
            # Data for display
            display_data = {
                "title": result["title"],
                "image_file": img_filename,
                "genres": ", ".join(result.get("genres", [])),
                "mood": result.get("mood", ""),
                "energy": result.get("energy_level", ""),
                "playlist_name": result.get("item_name", "Your Music"),
                "found_genres": bool(result.get("genres", [])),
                "genres_chart": genres_chart,
                "genre_percentages": genre_percentages,
                "playlist_url": playlist_url,
                "user_mood": user_mood
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
        "image_model_loaded": pipe is not None,
        "title_model_loaded": title_generator is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })

@app.route("/api/generate", methods=["POST"])
def api_generate():
    """API endpoint to generate covers programmatically"""
    try:
        data = request.json
        if not data or "spotify_url" not in data:
            return jsonify({"error": "Missing spotify_url in request"}), 400
            
        spotify_url = data.get("spotify_url")
        user_mood = data.get("mood", "")
        
        # Generate the cover
        result = generate_cover(spotify_url, user_mood)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
            
        # Return result data
        return jsonify({
            "success": True,
            "title": result["title"],
            "image_path": result["output_path"],
            "image_url": f"/generated_covers/{os.path.basename(result['output_path'])}",
            "data_file": result.get("data_file"),
            "genres": result.get("genres", []),
            "mood": result.get("mood", ""),
            "energy_level": result.get("energy_level", ""),
            "playlist_name": result.get("item_name", "")
        })
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/regenerate", methods=["POST"])
def api_regenerate():
    """API endpoint to regenerate cover art with the same playlist"""
    try:
        data = request.json
        if not data or "playlist_url" not in data:
            return jsonify({"error": "Missing playlist_url in request"}), 400
            
        spotify_url = data.get("playlist_url")
        user_mood = data.get("mood", "")
        
        # Generate a new seed to ensure variation
        random_seed = random.randint(1, 1000000)
        torch.manual_seed(random_seed)
        
        # Generate the cover
        result = generate_cover(spotify_url, user_mood)
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
            
        # Return result data
        return jsonify({
            "success": True,
            "title": result["title"],
            "image_path": result["output_path"],
            "image_url": f"/generated_covers/{os.path.basename(result['output_path'])}",
            "data_file": result.get("data_file")
        })
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import sys
    
    # Check if running in CLI mode
    if len(sys.argv) > 1:
        if sys.argv[1] == "--generate" and len(sys.argv) >= 3:
            print(f"Starting Spotify Cover Generator in CLI mode")
            print(f"Device: {device}")
            if torch.cuda.is_available():
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("Running on CPU (image generation will be slow)")
            
            initialize_models()
            
            spotify_url = sys.argv[2]
            mood = sys.argv[3] if len(sys.argv) >= 4 else None
            
            print(f"Generating cover for: {spotify_url}")
            if mood:
                print(f"Using mood: {mood}")
                
            result = generate_cover(spotify_url, mood)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                sys.exit(1)
                
            print(f"\nGeneration complete!")
            print(f"Title: {result['title']}")
            print(f"Image saved to: {result['output_path']}")
            print(f"Data saved to: {result.get('data_file', 'Not saved')}")
            sys.exit(0)
            
        elif sys.argv[1] == "--help":
            print("Spotify Cover Generator CLI Usage:")
            print("  Generate a cover: python app.py --generate <spotify_url> [mood]")
            print("  Start web server: python app.py")
            print("  Show this help:   python app.py --help")
            sys.exit(0)
    
    # Default to web server mode
    print(f"Starting Spotify Cover Generator")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (image generation will be slow)")
    
    initialize_models()
    app.run(debug=False, host="0.0.0.0", port=50)