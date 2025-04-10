document.addEventListener('DOMContentLoaded', function() {
    // Add visual feedback when pasting Spotify URL
    const playlistUrlInput = document.getElementById('playlist_url');
    if (playlistUrlInput) {
        // Check if input has value when page loads (e.g. after error)
        if (playlistUrlInput.value.trim() !== '') {
            playlistUrlInput.classList.add('has-value');
        }
        
        // Add validation for Spotify URLs
        playlistUrlInput.addEventListener('input', function() {
            const value = this.value.trim();
            if (value !== '') {
                this.classList.add('has-value');
                // Simple validation for Spotify playlist URL format
                if (value.includes('open.spotify.com/playlist/')) {
                    this.classList.remove('invalid');
                    this.classList.add('valid');
                } else {
                    this.classList.remove('valid');
                    this.classList.add('invalid');
                }
            } else {
                this.classList.remove('has-value');
                this.classList.remove('valid');
                this.classList.remove('invalid');
            }
        });
    }
    
    // Add loading state to form submission
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function() {
            const submitButton = this.querySelector('.submit-btn');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerHTML = 'Generating... <span class="spinner"></span>';
                submitButton.classList.add('loading');
            }
            
            // Create a loading overlay
            const loadingOverlay = document.createElement('div');
            loadingOverlay.className = 'loading-overlay';
            loadingOverlay.innerHTML = `
                <div class="loading-content">
                    <div class="loading-spinner"></div>
                    <p>Analyzing playlist and generating cover art...</p>
                    <p class="loading-subtext">This may take a minute or two</p>
                </div>
            `;
            document.body.appendChild(loadingOverlay);
            
            // Allow form submission to continue
            return true;
        });
    }
    
    // On result page, add animation for cover reveal
    const albumCover = document.querySelector('.album-cover');
    if (albumCover) {
        // Add fade-in animation when image loads
        albumCover.style.opacity = '0';
        albumCover.addEventListener('load', function() {
            setTimeout(() => {
                albumCover.style.transition = 'opacity 1s ease-in-out';
                albumCover.style.opacity = '1';
            }, 300);
        });
    }
    
    // Add CSS styles for the newly added elements
    const style = document.createElement('style');
    style.textContent = `
        input.valid {
            border-left: 4px solid #1DB954 !important;
        }
        
        input.invalid {
            border-left: 4px solid #ff5252 !important;
        }
        
        .submit-btn.loading {
            background-color: #168f40;
            pointer-events: none;
        }
        
        .spinner {
            display: inline-block;
            width: 15px;
            height: 15px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
            margin-left: 8px;
            vertical-align: middle;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        
        .loading-content {
            text-align: center;
            padding: 30px;
            background: #1e1e1e;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }
        
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 5px solid rgba(29, 185, 84, 0.3);
            border-radius: 50%;
            border-top-color: #1DB954;
            margin: 0 auto 20px;
            animation: spin 1s ease infinite;
        }
        
        .loading-subtext {
            color: #a0a0a0;
            font-size: 0.9rem;
        }
    `;
    document.head.appendChild(style);
});