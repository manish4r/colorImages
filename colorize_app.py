# NOTE: This must be the first call in order to work properly!
from deoldify import device
from deoldify.device_id import DeviceId
import torch
import warnings
from os import path, makedirs
# import fastai
from deoldify.visualize import *
import requests  # Added for robust model downloading
import gradio as gr # Added for the UI
import time

# --- 1. Setup Device ---
# Set device to GPU0 if available, otherwise fall back to CPU
try:
    device.set(device=DeviceId.GPU0)
    if not torch.cuda.is_available():
        print('GPU not available. Falling back to CPU.')
        device.set(device=DeviceId.CPU)
except Exception as e:
    print(f'Error setting device: {e}. Falling back to CPU.')
    device.set(device=DeviceId.CPU)

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

# --- 2. Download Model (if needed) ---
def download_model():
    """Checks for the model file and downloads it if missing."""
    model_dir = 'models'
    model_name = 'ColorizeArtistic_gen.pth'
    model_path = path.join(model_dir, model_name)
    model_url = 'https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth'

    if not path.exists(model_path):
        print(f'Model not found at {model_path}. Downloading...')
        try:
            makedirs(model_dir, exist_ok=True)
            # Use requests to download
            with requests.get(model_url, stream=True) as r:
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print('Model downloaded successfully.')
        except Exception as e:
            print(f'Error downloading model: {e}')
            print('Please download the model manually from:')
            print(f'{model_url}')
            print(f'And place it in: {model_path}')
            raise
    else:
        print('Model already exists.')

# Run the model download check
download_model()

# --- 3. Load the Colorizer ---
# This is now global so the interface function can access it
try:
    colorizer = get_image_colorizer(artistic=True)
    print("Colorizer loaded successfully.")
except Exception as e:
    print(f"Fatal error loading colorizer: {e}")
    print("Please ensure the model file is valid and in the './models/' directory.")
    exit() # Exit if we can't load the model

# --- 4. Define the Interface Function ---
def colorize_from_url(image_url, render_factor, watermarked):
    """
    This function will be called by the Gradio interface.
    It takes the inputs from the UI, runs the colorizer,
    and returns the path to the output image and a status message.
    """
    if not image_url:
        return None, "Error: Please enter a URL."
    
    start_time = time.time()
    print(f"Processing {image_url}...")
    try:
        # This function downloads the image and saves the result
        image_path = colorizer.plot_transformed_image_from_url(
        url=image_url,
        render_factor=int(render_factor), # Ensure render_factor is int
        # compare=compare_view,
        watermarked=watermarked
        )
        # Return the path to the generated image
        end_time = time.time()
        duration = end_time - start_time
        return image_url,image_path, f"Successfully colorized {image_url} in {duration:.2f} seconds."
    except Exception as e:
        print(f"Error processing image: {e}")
        return None,None, f"Error: {str(e)}. Check if the URL is a valid image."

# --- 5. Create and Launch the Gradio Interface ---

# Set default values from your original script
default_url = 'https://images.pexels.com/photos/276374/pexels-photo-276374.jpeg'
default_render = 35
default_watermarked = True

# Build the interface
iface = gr.Interface(
    flagging_mode="never",  # This removes the Flag button
    fn=colorize_from_url,
    inputs=[
        gr.Textbox(label="Image URL", value=default_url),
        gr.Slider(minimum=5, maximum=40, value=default_render, step=1, label="Render Factor"),
        gr.Checkbox(value=default_watermarked, label="Add DeOldify Watermark"),
        # gr.Checkbox(value=True, label="Show Before/After Comparison")
    ],
    outputs=[
        gr.Image(type="filepath", label="Original Image"),
        gr.Image(type="filepath", label="Colorized Image"),
        gr.Textbox(label="Status")
    ],
    title="DeOldify Image Colorizer 🎨",
    description="Enter the URL of a black and white image to colorize it. The 'Artistic' model will be used."
)

# Launch the app!
if __name__ == "__main__":
    print("Launching Gradio interface...")
    print("Open the following URL in your browser:")
    iface.launch()
