import sys
import os
import json
import random
import string
import statistics
from pathlib import Path
from io import BytesIO
from collections import Counter

from flask import Flask, request, render_template, send_from_directory
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import torch
from diffusers import StableDiffusionXLPipeline
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

# Initialize Spotify client with better error handling
print("Initializing Spotify client...")
try:
    # First, verify that credentials are loaded correctly
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("ERROR: Spotify credentials are missing or empty")
        sp = None
    else:
        print(f"Using Client ID: {SPOTIFY_CLIENT_ID[:5]}...{SPOTIFY_CLIENT_ID[-5:]} (showing first/last 5 chars only)")
        
        # Create auth manager with explicit caching disabled to force fresh tokens
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            cache_handler=None  # Disable caching to force new token
        )
        
        # Create the Spotify client with reduced complexity first
        sp = spotipy.Spotify(
            auth_manager=auth_manager,
            requests_timeout=30
        )
        
        # Test the connection with a simple API call that doesn't require user authentication
        print("Testing Spotify API connection...")
        test_result = sp.search(q='test', limit=1)
        
        if test_result and 'tracks' in test_result:
            print("✓ Spotify API connection successful")
        else:
            print("✗ Spotify API test failed - no tracks returned")
            sp = None
except Exception as e:
    print(f"✗ Error initializing Spotify API: {str(e)}")
    # Get more detailed error information
    import traceback
    traceback.print_exc()
    sp = None

# Create a torch device object (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("CUDA is available! Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Warning: CUDA not available. Using CPU.")

# Initialize the stable diffusion pipeline
print("Loading Stable Diffusion XL model...")
model_name = "stabilityai/stable-diffusion-xl-base-1.0"
model_cache_dir = CACHE_DIR / model_name.replace("/", "_")
model_cache_dir.mkdir(exist_ok=True)

# Create the pipeline variable
pipe = None

# Load the pipeline with error handling
try:
    if device.type == "cuda":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16",  
            use_safetensors=True,
            cache_dir=model_cache_dir
        ).to(device)
        
        # Enable memory optimizations
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except Exception as e:
            print(f"Could not enable xformers: {e}")
            
        # Verify the model loaded correctly
        print("Model loaded successfully")
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            use_safetensors=True,
            cache_dir=model_cache_dir
        ).to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to a simpler model if needed
    model_name = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR / model_name.replace("/", "_")
    ).to(device)

# Initialize the title generator
print("Loading title generator model...")
device_id = 0 if device.type == "cuda" else -1
title_generator = text_pipeline(
    "text-generation",
    model="EleutherAI/gpt-j-6B",
    device=device_id,
    model_kwargs={"cache_dir": CACHE_DIR / "title_generator"}
)

# Assume templates already exist
TEMPLATES_DIR = BASE_DIR / "templates"
app = Flask(__name__, template_folder=str(TEMPLATES_DIR))

def generate_random_string(size=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))

def extract_playlist_data(playlist_url):
    """
    Extract comprehensive data from the playlist including:
    - Genres
    - Audio features (danceability, energy, etc.)
    - Track metadata
    
    Returns a dictionary with all playlist data.
    """
    if not sp:
        return {"error": "Spotify API client not initialized. Check your API credentials."}
        
    # Handle different Spotify URL formats
    if "playlist/" in playlist_url:
        playlist_id = playlist_url.split("playlist/")[-1].split("?")[0].split("/")[0]
    else:
        return {"error": "Invalid Spotify playlist URL format"}
    
    print(f"Extracting data for playlist ID: {playlist_id}")
    
    # Get playlist tracks
    try:
        # Get basic playlist info first
        try:
            playlist_info = sp.playlist(playlist_id, fields="name,description")
            playlist_name = playlist_info.get("name", "Unknown Playlist")
            print(f"Found playlist: {playlist_name}")
        except Exception as e:
            print(f"Warning: Couldn't get playlist info: {e}")
            playlist_name = "Unknown Playlist"
            
        # Then get the tracks
        results = sp.playlist_tracks(
            playlist_id, 
            fields="items(track(id,name,artists(id,name))),next",
            market="US"  # Adding market parameter to avoid region restrictions
        )
        tracks = results.get("items", [])
        
        # Get more tracks if pagination exists (limit to 50 tracks for performance)
        while results.get('next') and len(tracks) < 50:
            results = sp.next(results)
            tracks.extend(results.get("items", []))
            
        if not tracks:
            return {"error": "No tracks found in playlist"}
            
        print(f"Found {len(tracks)} tracks in playlist")
            
        # Extract track IDs for audio features
        track_ids = []
        artists = []
        track_names = []
        
        for item in tracks:
            track = item.get("track")
            if track and track.get("id"):  # Only process tracks with valid IDs
                track_ids.append(track.get("id"))
                track_names.append(track.get("name"))
                if track.get("artists"):
                    for artist in track.get("artists"):
                        if artist.get("id"):  # Only add valid artist IDs
                            artists.append(artist.get("id"))
        
        if not track_ids:
            return {
                "error": "No valid tracks found in playlist. This may be due to regional restrictions."
            }
            
        # Fallback data in case API calls fail
        fallback_data = {
            "track_names": track_names[:10],  # First 10 track names
            "genres": ["unknown"],
            "audio_features": {
                "danceability": 0.5, "energy": 0.5, "loudness": -10,
                "speechiness": 0.1, "acousticness": 0.5, "instrumentalness": 0.1,
                "liveness": 0.1, "valence": 0.5, "tempo": 120
            },
            "tempo_range": "moderate",
            "energy_level": "balanced",
            "mood_descriptor": "balanced",
            "playlist_name": playlist_name
        }
        
        # Get audio features in smaller batches (20 at a time to avoid API limits)
        audio_features = []
        
        # Use smaller batch size and add delay to avoid hitting rate limits
        batch_size = 20
        for i in range(0, len(track_ids), batch_size):
            try:
                batch = track_ids[i:i+batch_size]
                if batch:
                    print(f"Fetching audio features for batch {i//batch_size + 1}/{(len(track_ids) + batch_size - 1)//batch_size}")
                    features = sp.audio_features(batch)
                    if features:
                        valid_features = [f for f in features if f]
                        print(f"Got {len(valid_features)} valid features")
                        audio_features.extend(valid_features)
                    # Small delay to avoid rate limiting
                    import time
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error fetching audio features batch: {e}")
        
        # If we couldn't get audio features, use fallback
        if not audio_features:
            print("Warning: Could not get audio features, using fallback data")
            return fallback_data
        
        # Get artist genres (in smaller batches)
        genres_counter = Counter()
        batch_size = 10  # Smaller batch size
        
        for i in range(0, len(artists), batch_size):
            try:
                batch = list(set(artists[i:i+batch_size]))  # Remove duplicates
                if batch:
                    print(f"Fetching artist data for batch {i//batch_size + 1}/{(len(artists) + batch_size - 1)//batch_size}")
                    artists_data = sp.artists(batch)
                    if artists_data and "artists" in artists_data:
                        for artist in artists_data["artists"]:
                            for genre in artist.get("genres", []):
                                genres_counter[genre] += 1
                    # Small delay to avoid rate limiting
                    import time
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error fetching artist genres batch: {e}")
        
        # Calculate average audio features
        avg_features = {}
        if audio_features:
            feature_keys = [
                "danceability", "energy", "loudness", "speechiness", 
                "acousticness", "instrumentalness", "liveness", "valence", "tempo"
            ]
            
            for key in feature_keys:
                try:
                    values = [float(af[key]) for af in audio_features if key in af and af[key] is not None]
                    if values:
                        avg_features[key] = statistics.mean(values)
                    else:
                        avg_features[key] = fallback_data["audio_features"][key]
                except Exception as e:
                    print(f"Error calculating average for {key}: {e}")
                    avg_features[key] = fallback_data["audio_features"][key]
        
        # Get top genres (most frequent)
        top_genres = [genre for genre, _ in genres_counter.most_common(10)]
        
        # If no genres found, provide generic ones
        if not top_genres:
            top_genres = ["music", "contemporary", "playlist"]
        
        return {
            "track_names": track_names[:10],  # First 10 track names
            "genres": top_genres,
            "audio_features": avg_features,
            "tempo_range": get_tempo_range(avg_features.get("tempo", 0)),
            "energy_level": get_energy_level(avg_features.get("energy", 0)),
            "mood_descriptor": get_mood_descriptor(avg_features),
            "playlist_name": playlist_name
        }
    
    except Exception as e:
        print(f"Error extracting playlist data: {e}")
        return {"error": f"Error accessing Spotify API: {str(e)}. Please check your API credentials and try again."}

def extract_playlist_data_safe(playlist_url, spotify_client):
    """
    A version of extract_playlist_data that uses a provided Spotify client
    instead of the global one. Used for fallback when global client fails.
    """
    # Handle different Spotify URL formats
    if "playlist/" in playlist_url:
        playlist_id = playlist_url.split("playlist/")[-1].split("?")[0].split("/")[0]
    else:
        return {"error": "Invalid Spotify playlist URL format"}
    
    print(f"Extracting data for playlist ID: {playlist_id} with dedicated client")
    
    # Get playlist tracks
    try:
        # Get basic playlist info first
        try:
            playlist_info = spotify_client.playlist(playlist_id, fields="name,description")
            playlist_name = playlist_info.get("name", "Unknown Playlist")
            print(f"Found playlist: {playlist_name}")
        except Exception as e:
            print(f"Warning: Couldn't get playlist info: {e}")
            playlist_name = "Unknown Playlist"
            
        # Then get the tracks
        results = spotify_client.playlist_tracks(
            playlist_id, 
            fields="items(track(id,name,artists(id,name))),next",
            market="US"  # Adding market parameter to avoid region restrictions
        )
        tracks = results.get("items", [])
        
        # Only get first page for speed (up to 100 tracks)
        if not tracks:
            return {"error": "No tracks found in playlist"}
            
        print(f"Found {len(tracks)} tracks in playlist")
            
        # Extract track IDs for audio features
        track_ids = []
        artists = []
        track_names = []
        
        for item in tracks:
            track = item.get("track")
            if track and track.get("id"):  # Only process tracks with valid IDs
                track_ids.append(track.get("id"))
                track_names.append(track.get("name"))
                if track.get("artists"):
                    for artist in track.get("artists"):
                        if artist.get("id"):  # Only add valid artist IDs
                            artists.append(artist.get("id"))
        
        if not track_ids:
            return {
                "error": "No valid tracks found in playlist. This may be due to regional restrictions."
            }
        
        # Get audio features in smaller batches (20 at a time to avoid API limits)
        audio_features = []
        
        # Use smaller batch size and add delay to avoid hitting rate limits
        batch_size = 20
        for i in range(0, min(len(track_ids), 40), batch_size):  # Limit to 40 tracks max for speed
            try:
                batch = track_ids[i:i+batch_size]
                if batch:
                    print(f"Fetching audio features for batch {i//batch_size + 1}")
                    features = spotify_client.audio_features(batch)
                    if features:
                        valid_features = [f for f in features if f]
                        print(f"Got {len(valid_features)} valid features")
                        audio_features.extend(valid_features)
            except Exception as e:
                print(f"Error fetching audio features batch: {e}")
                # Continue with what we have
                break
        
        # Just get a few genres from a few artists for speed
        genres = []
        try:
            # Get just a few artists (max 5)
            sample_artists = list(set(artists))[:5]
            if sample_artists:
                artists_data = spotify_client.artists(sample_artists)
                if artists_data and "artists" in artists_data:
                    for artist in artists_data["artists"]:
                        genres.extend(artist.get("genres", []))
        except Exception as e:
            print(f"Error fetching artist genres: {e}")
        
        # Deduplicate genres
        unique_genres = list(set(genres))
        
        # Calculate average audio features
        avg_features = {}
        if audio_features:
            feature_keys = [
                "danceability", "energy", "loudness", "speechiness", 
                "acousticness", "instrumentalness", "liveness", "valence", "tempo"
            ]
            
            for key in feature_keys:
                try:
                    values = [float(af[key]) for af in audio_features if key in af and af[key] is not None]
                    if values:
                        avg_features[key] = statistics.mean(values)
                    else:
                        avg_features[key] = 0.5
                except Exception as e:
                    print(f"Error calculating average for {key}: {e}")
                    avg_features[key] = 0.5
        else:
            # Fallback defaults
            avg_features = {
                "danceability": 0.5, "energy": 0.5, "loudness": -10,
                "speechiness": 0.1, "acousticness": 0.5, "instrumentalness": 0.1,
                "liveness": 0.1, "valence": 0.5, "tempo": 120
            }
        
        # If no genres found, provide generic ones
        if not unique_genres:
            unique_genres = ["music", "contemporary", "playlist"]
        
        return {
            "track_names": track_names[:10],  # First 10 track names
            "genres": unique_genres[:10],     # Up to 10 genres
            "audio_features": avg_features,
            "tempo_range": get_tempo_range(avg_features.get("tempo", 0)),
            "energy_level": get_energy_level(avg_features.get("energy", 0)),
            "mood_descriptor": get_mood_descriptor(avg_features),
            "playlist_name": playlist_name
        }
    
    except Exception as e:
        print(f"Error extracting playlist data with dedicated client: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a basic dataset we can work with
        return {
            "track_names": ["Track"],
            "genres": ["music", "contemporary"],
            "audio_features": {
                "danceability": 0.5, "energy": 0.5, "loudness": -10,
                "speechiness": 0.1, "acousticness": 0.5, "instrumentalness": 0.1,
                "liveness": 0.1, "valence": 0.5, "tempo": 120
            },
            "tempo_range": "moderate",
            "energy_level": "balanced",
            "mood_descriptor": "balanced",
            "playlist_name": "Playlist"
        }

def get_tempo_range(tempo):
    """Convert numerical tempo to descriptive range"""
    if tempo < 80:
        return "slow"
    elif tempo < 120:
        return "moderate"
    else:
        return "fast"

def get_energy_level(energy):
    """Convert numerical energy to descriptive level"""
    if energy < 0.33:
        return "calm"
    elif energy < 0.66:
        return "balanced"
    else:
        return "energetic"

def get_mood_descriptor(features):
    """
    Generate mood description based on valence and energy
    Valence represents musical positiveness (high = happy, low = sad)
    Energy represents intensity and activity (high = energetic, low = calm)
    """
    valence = features.get("valence", 0.5)
    energy = features.get("energy", 0.5)
    
    # Mood matrix based on valence and energy
    if valence > 0.7 and energy > 0.7:
        return "euphoric"
    elif valence > 0.7 and energy < 0.3:
        return "peaceful"
    elif valence < 0.3 and energy > 0.7:
        return "angry"
    elif valence < 0.3 and energy < 0.3:
        return "melancholic"
    elif valence > 0.5 and energy > 0.5:
        return "upbeat"
    elif valence > 0.5 and energy < 0.5:
        return "relaxed"
    elif valence < 0.5 and energy > 0.5:
        return "tense"
    elif valence < 0.5 and energy < 0.5:
        return "somber"
    else:
        return "balanced"

def create_prompt_from_data(playlist_data, user_mood=None):
    """
    Create a detailed and specific prompt for the image generator
    based on the extracted playlist data and optional user mood.
    """
    genres_str = ", ".join(playlist_data.get("genres", ["various"]))
    tempo = playlist_data.get("tempo_range", "moderate")
    energy = playlist_data.get("energy_level", "balanced")
    mood = user_mood if user_mood else playlist_data.get("mood_descriptor", "balanced")
    
    # Get audio features for more detailed prompt
    features = playlist_data.get("audio_features", {})
    danceability = features.get("danceability", 0.5)
    acousticness = features.get("acousticness", 0.5)
    instrumentalness = features.get("instrumentalness", 0.5)
    
    # Build detailed prompt based on audio features
    prompt_elements = []
    
    # Add genre-based elements
    prompt_elements.append(f"An artistic album cover for {genres_str} music")
    
    # Add mood descriptor
    prompt_elements.append(f"with a {mood} atmosphere")
    
    # Add tempo and energy descriptors
    prompt_elements.append(f"that feels {tempo} and {energy}")
    
    # Add special elements based on audio features
    if danceability > 0.7:
        prompt_elements.append("with rhythmic patterns")
    if acousticness > 0.7:
        prompt_elements.append("with organic textures and natural elements")
    if instrumentalness > 0.7:
        prompt_elements.append("with abstract musical representations")
    
    # Add style descriptors for more interesting results
    style_descriptors = [
        "highly detailed", "artistic", "professional album artwork",
        "colorful", "trending on artstation"
    ]
    prompt_elements.append(", ".join(style_descriptors))
    
    # Combine all elements
    prompt = ", ".join(prompt_elements)
    
    return prompt

def clear_cuda_memory():
    """
    Aggressively clear CUDA memory to prevent out-of-memory errors.
    """
    if torch.cuda.is_available():
        print("Clearing CUDA memory...")
        # Clear PyTorch's cache
        torch.cuda.empty_cache()
        
        # If available, use NVIDIA tools for more aggressive clearing
        try:
            import gc
            gc.collect()
        except Exception as e:
            print(f"Warning: Could not perform garbage collection: {e}")

def generate_cover_image(prompt):
    """
    Generate a cover image using less memory.
    """
    global pipe
    
    print(f"Generating image with prompt: {prompt}")
    
    # Clear memory before generation
    clear_cuda_memory()
    
    # Enhanced negative prompt (shortened to use less memory)
    negative_prompt = "no humans, no faces, low quality, worst quality, blurry"
    
    try:
        # Use lower resolution and fewer steps to save memory
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,  # Reduced from 40
            guidance_scale=7.5,      # Slightly reduced
            width=768,              # Reduced from 1024
            height=768              # Reduced from 1024
        )
        
        image = result.images[0] if hasattr(result, 'images') else result[0]
        
        # Immediately clear memory after generation
        clear_cuda_memory()
            
        return image
        
    except Exception as e:
        print(f"Error generating image: {e}")
        
        # Try one more time with even lower settings
        try:
            print("Retrying with lower settings...")
            clear_cuda_memory()
            
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=20,
                guidance_scale=7.0,
                width=512,
                height=512
            )
            
            image = result.images[0] if hasattr(result, 'images') else result[0]
            clear_cuda_memory()
            return image
            
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            # Create a colored fallback image
            fallback = Image.new('RGB', (512, 512), color='#3A506B')
            return fallback
def generate_title(playlist_data, mood=""):
    """
    Generate a unique album title using the playlist data and mood.
    """
    genres_str = ", ".join(playlist_data.get("genres", ["various"]))
    tempo = playlist_data.get("tempo_range", "moderate")
    energy = playlist_data.get("energy_level", "balanced")
    detected_mood = playlist_data.get("mood_descriptor", "balanced")
    
    # Use user-specified mood if provided, otherwise use detected mood
    mood_to_use = mood if mood else detected_mood
    
    # Get a few track names to inspire the title
    track_names = playlist_data.get("track_names", [])
    track_sample = ", ".join(track_names[:3]) if track_names else ""
    
    prompt = (
        "Create a unique and evocative album title for a music playlist. "
        "The album should feel original and creative. "
        f"It has a {mood_to_use} mood, {tempo} tempo, and {energy} energy level. "
        f"It is characterized by the following genres: {genres_str}. "
    )
    
    if track_sample:
        prompt += f"Some example tracks include: {track_sample}. "
        
    prompt += "Do not repeat or list the genres in the title; instead, invent a new, abstract title. Title:"
    
    try:
        output = title_generator(
            prompt,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.8
        )
        full_text = output[0]['generated_text']
        if "Title:" in full_text:
            title = full_text.split("Title:")[1].strip()
        else:
            title = full_text.strip()
        return title.split("\n")[0][:50]  # Limit to first 50 characters
    except Exception as e:
        print(f"Error generating title: {e}")
        # Fallback title generation if the model fails
        moods = ["Echoes", "Waves", "Pulse", "Drift", "Glow", "Haze"]
        adjectives = ["Infinite", "Electric", "Cosmic", "Radiant", "Velvet", "Crystal"]
        return f"{random.choice(adjectives)} {random.choice(moods)}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        playlist_url = request.form.get("playlist_url")
        user_mood = request.form.get("mood", "").strip()
        
        if not playlist_url:
            return render_template("index.html", error="Please enter a Spotify playlist URL.")
            
        print(f"Processing playlist: {playlist_url}")
        
        # Check if Spotify API client is initialized
        if sp is None:
            # Try to reinitialize the client
            print("Attempting to reinitialize Spotify client...")
            try:
                auth_manager = SpotifyClientCredentials(
                    client_id=SPOTIFY_CLIENT_ID,
                    client_secret=SPOTIFY_CLIENT_SECRET,
                    cache_handler=None  # Disable caching
                )
                
                # Create a fresh client for this request
                temp_sp = spotipy.Spotify(
                    auth_manager=auth_manager,
                    requests_timeout=30
                )
                
                # Test with a simple search
                test_result = temp_sp.search(q='test', limit=1)
                if not test_result or 'tracks' not in test_result:
                    raise Exception("API test failed")
                
                # If we get here, we have a working client
                print("Successfully reinitialized Spotify client for this request")
                
                # Continue with the request using the temporary client
                playlist_data = extract_playlist_data_safe(playlist_url, temp_sp)
            except Exception as e:
                print(f"Reinitialization failed: {e}")
                return render_template("index.html", 
                    error="Could not connect to Spotify API. Please check your network connection and API credentials.")
        else:
            # Use the global client
            playlist_data = extract_playlist_data(playlist_url)
        
        if "error" in playlist_data:
            return render_template("index.html", error=f"Error: {playlist_data['error']}")
        
        # Create image prompt from playlist data
        image_prompt = create_prompt_from_data(playlist_data, user_mood)
        print(f"Generated prompt: {image_prompt}")
        
        # Verify the image generation pipeline is loaded
        if pipe is None:
            return render_template("index.html", error="Image generation model not loaded properly. Please restart the application.")
        
        try:
            # Generate cover image
            cover_image = generate_cover_image(image_prompt)
            
            # Save to the covers directory with a random filename
            img_filename = generate_random_string() + ".png"
            img_path = COVERS_DIR / img_filename
            print(f"Saving image to {img_path}")
            cover_image.save(img_path)
            
            # Generate title
            title = generate_title(playlist_data, user_mood)
            print(f"Generated title: {title}")
            
            # Prepare data for display
            display_data = {
                "title": title,
                "image_file": img_filename,
                "genres": ", ".join(playlist_data.get("genres", [])),
                "mood": user_mood if user_mood else playlist_data.get("mood_descriptor", ""),
                "tempo": playlist_data.get("tempo_range", ""),
                "energy": playlist_data.get("energy_level", ""),
                "features": playlist_data.get("audio_features", {}),
                "playlist_name": playlist_data.get("playlist_name", "Your Playlist")
            }
            
            return render_template("result.html", **display_data)
        except Exception as e:
            print(f"Error in generation process: {e}")
            return render_template("index.html", error=f"Error generating cover: {str(e)}")
    else:
        # Check if Spotify API credentials are available
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            return render_template("index.html", error="Spotify API credentials not found. Please set them in the .env file.")
        
        return render_template("index.html")

@app.route("/generated_covers/<path:filename>")
def serve_image(filename):
    return send_from_directory(COVERS_DIR, filename)

if __name__ == "__main__":
    print("Starting Flask server on port 50")
    app.run(debug=True, host="0.0.0.0", port=50)