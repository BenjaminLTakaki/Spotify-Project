# Spotify Playlist Cover Generator

An AI-powered web application that analyzes Spotify playlists and generates custom album artwork and titles based on the musical genres detected.

## Features

### Core Features
- Analyzes Spotify playlists or albums to extract artist genres
- Generates custom album covers using Stable Diffusion
- Creates unique album titles with Meta-Llama 3
- Visualizes genre distribution with interactive charts

### Enhanced Features
- **Preset Style Selection**: Quick buttons for different visual styles (Minimalist, High Contrast, Retro, Bold Colors)
- **Genre Percentage Visualization**: Visual breakdown of genres with colored progress bars
- **Cover Regeneration**: Regenerate new covers with the same playlist without starting over
- **Copy Functionality**: Download generated covers and copy titles with a single click
- **Loading States**: Visual feedback during the generation process

## Requirements
- Python 3.8+
- Flask
- Spotipy (Spotify API client)
- PyTorch
- Diffusers (for Stable Diffusion)
- Transformers (for Meta-Llama 3)

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Spotify API credentials:
   ```
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   ```
4. Run the application:
   ```
   python app.py
   ```

## Usage

1. Enter a Spotify playlist or album URL
2. (Optional) Select a preset style or enter a custom mood
3. Click "Generate Cover" and wait for the magic to happen
4. View your custom album cover, title, and genre analysis
5. Use the "Regenerate Cover" button to create variations with the same playlist
6. Download your cover or copy the title to use elsewhere

## How It Works

1. **Data Extraction**: Analyzes Spotify playlist data to identify genres
2. **Prompt Creation**: Builds intelligent prompts based on genre analysis
3. **Image Generation**: Uses Stable Diffusion to create a custom album cover
4. **Title Generation**: Uses Meta-Llama 3 to generate a fitting album title
5. **Visualization**: Creates charts and displays genre distribution

## License
MIT License
