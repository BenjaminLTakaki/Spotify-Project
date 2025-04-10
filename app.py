import os, sys, json, random, string, time, gc, logging
from pathlib import Path
from io import BytesIO
from flask import Flask, request, render_template, send_from_directory, url_for
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from transformers import pipeline as text_pipeline
from PIL import Image
from dotenv import load_dotenv

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create cache directory
CACHE_DIR = Path("./model_cache")
CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)

# Load environment variables
load_dotenv()
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise ValueError("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in the .env file.")

# Initialize Spotify client
sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET
    ),
    requests_timeout=15
)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Model selection based on memory
DEFAULT_MODEL = "CompVis/stable-diffusion-v1-4"  # Smaller model
HIGH_QUALITY_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # Larger model

# Memory utility functions
def clear_memory():
    """Clear CUDA cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_available_memory():
    """Get available CUDA memory in GB"""
    if torch.cuda.is_available():
        return (torch.cuda.get_device_properties(0).total_memory - 
                torch.cuda.memory_allocated(0)) / 1024**3
    return 0

def load_diffusion_pipeline(high_quality=False):
    """Load the diffusion pipeline with memory optimization"""
    logger.info("Loading diffusion pipeline...")
    start_time = time.time()
    clear_memory()
    
    # Select model based on available memory
    available_mem = get_available_memory()
    use_high_quality = high_quality and available_mem > 10.0
    model_name = HIGH_QUALITY_MODEL if use_high_quality else DEFAULT_MODEL
    logger.info(f"Selected model: {model_name} (Memory: {available_mem:.2f} GB)")
    
    # Create model directory
    model_cache_dir = CACHE_DIR / model_name.replace("/", "_")
    model_cache_dir.mkdir(exist_ok=True)
    
    try:
        if device.type == "cuda":
            # Use float16 for GPU
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                variant="fp16" if "xl" in model_name else None,
                use_safetensors=True,
                cache_dir=model_cache_dir,
                resume_download=True,
                local_files_only=False,
                safety_checker=None  # Disable safety checker to save memory
            ).to(device)
            
            # Memory optimizations
            if available_mem < 6.0:
                logger.info("Enabling CPU offloading")
                pipe.enable_sequential_cpu_offload()
            pipe.enable_attention_slicing(slice_size="auto")
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
        else:
            # Use float32 for CPU
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                use_safetensors=True,
                cache_dir=model_cache_dir,
                resume_download=True,
                safety_checker=None
            ).to("cpu")
        
        end_time = time.time()
        logger.info(f"Pipeline loaded in {end_time - start_time:.2f} seconds")
        return pipe
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory. Trying with smaller model and optimizations...")
        clear_memory()
        
        # Retry with minimal model
        pipe = StableDiffusionPipeline.from_pretrained(
            DEFAULT_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=CACHE_DIR / DEFAULT_MODEL.replace("/", "_"),
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        pipe.enable_sequential_cpu_offload()
        pipe.enable_attention_slicing(slice_size=1)
        return pipe
    
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        # If all else fails, try CPU
        if device.type == "cuda":
            logger.info("Attempting CPU fallback")
            return load_diffusion_pipeline_cpu()
        raise

def load_diffusion_pipeline_cpu():
    """Load the pipeline on CPU as a last resort"""
    return StableDiffusionPipeline.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.float32,
        use_safetensors=True,
        cache_dir=CACHE_DIR / DEFAULT_MODEL.replace("/", "_"),
        safety_checker=None
    ).to("cpu")

def load_title_generator():
    """Load the title generator pipeline"""
    try:
        device_id = 0 if device.type == "cuda" else -1
        return text_pipeline(
            "text-generation",
            model="EleutherAI/gpt-j-6B",
            device=device_id,
            model_kwargs={"cache_dir": CACHE_DIR / "title_generator"}
        )
    except Exception as e:
        logger.error(f"Error loading title generator: {e}")
        # Return a simple fallback function
        def simple_title_generator(prompt, **kwargs):
            """Fallback title generator that doesn't require a model"""
            words = ['Sonic', 'Rhythmic', 'Melodic', 'Harmonic', 'Electric', 'Vibrant']
            nouns = ['Journey', 'Waves', 'Paths', 'Horizons', 'Dreams', 'Echoes']
            title = f"{random.choice(words)} {random.choice(nouns)}"
            return [{'generated_text': f"Title: {title}"}]
        return simple_title_generator

def generate_random_string(size=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))

def extract_playlist_genres(playlist_url):
    """Extract genres from a Spotify playlist"""
    playlist_id = playlist_url.split("/")[-1].split("?")[0]
    results = sp.playlist_tracks(playlist_id)
    genre_set = set()
    for item in results.get("items", []):
        track = item.get("track")
        if track and track.get("artists"):
            artist_info = sp.artist(track["artists"][0]["id"])
            for genre in artist_info.get("genres", []):
                genre_set.add(genre)
    return ", ".join(genre_set) if genre_set else "various genres"

def generate_cover_image(prompt, low_memory=False):
    """Generate a cover image with memory optimization"""
    clear_memory()
    
    # Enhanced negative prompt to ensure no human figures
    negative_prompt = (
        "no humans, no faces, no body parts, no humanoid figures, no people, "
        "no portraits, no crowds, no human-like creatures, no person, no man, no woman, "
        "no child, no hands, no fingers, avoid any recognizable human features, "
        "no human subjects, no human elements, no human presence, no human forms"
    )
    
    steps = 20 if low_memory else 30
    
    try:
        logger.info(f"Generating image with prompt: {prompt}")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
        )
        image = result.images[0]
        clear_memory()
        return image
    
    except torch.cuda.OutOfMemoryError:
        clear_memory()
        # Try with minimal settings
        logger.info("Out of memory. Trying with minimal settings...")
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=15,
                guidance_scale=5.0,
            )
            return result.images[0]
        except Exception:
            # Create a black image as fallback
            img = Image.new('RGB', (512, 512), color='black')
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.text((50, 240), "Error generating image", fill="white")
            return img
    
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        img = Image.new('RGB', (512, 512), color='black')
        return img

def generate_title(genres, mood=""):
    """Generate a playlist title"""
    prompt = "Create a unique album title for a playlist. "
    if mood:
        prompt += f"It has a {mood} mood, "
    prompt += f"with genres: {genres}. Title:"
    
    try:
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
        return title.split("\n")[0][:50]
    except Exception as e:
        logger.error(f"Error generating title: {e}")
        return f"Playlist: {genres[:30]}"

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            playlist_url = request.form.get("playlist_url")
            mood = request.form.get("mood", "").strip()
            low_memory = get_available_memory() < 6.0
            
            if not playlist_url:
                return render_template("index.html", error="Please enter a Spotify playlist URL.")
            
            genres = extract_playlist_genres(playlist_url)
            
            image_prompt = f"An artistic album cover for a playlist"
            if mood:
                image_prompt += f" with a {mood} mood"
            image_prompt += f", inspired by genres: {genres}"
            
            cover_image = generate_cover_image(image_prompt, low_memory)
            
            # Create generated_covers directory if it doesn't exist
            output_dir = Path("./generated_covers")
            output_dir.mkdir(exist_ok=True)
            
            # Generate a random filename
            random_filename = generate_random_string() + ".png"
            img_path = output_dir / random_filename
            
            # Save the image with full path
            logger.info(f"Saving image to {img_path}")
            cover_image.save(img_path)
            
            # Set the relative path for the template
            relative_path = f"generated_covers/{random_filename}"
            
            title = generate_title(genres, mood)
            
            return render_template(
                "result.html",
                title=title,
                image_file=relative_path,
                playlist_keywords=genres,
                mood=mood
            )
        except Exception as e:
            logger.error(f"Error: {e}")
            return render_template("index.html", error=f"An error occurred: {str(e)}")
    else:
        return render_template("index.html")

@app.route("/generated_covers/<path:filename>")
def serve_image(filename):
    """Serve images from the generated_covers directory"""
    return send_from_directory('generated_covers', filename)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Spotify Playlist Cover Generator')
    parser.add_argument('--high-quality', action='store_true', help='Use high-quality model if possible')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU mode')
    parser.add_argument('--port', type=int, default=50, help='Port number')
    
    args = parser.parse_args()
    
    if args.cpu_only:
        device = torch.device("cpu")
    
    # Load models
    pipe = load_diffusion_pipeline(high_quality=args.high_quality)
    title_generator = load_title_generator()
    
    app.run(debug=True, host="0.0.0.0", port=args.port)