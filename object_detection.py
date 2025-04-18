import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import tempfile
import subprocess
import sys
import gradio


def ensure_package_installed(package_name):
    try:
        __import__(package_name)
    except ImportError:
        print(f"{package_name} package not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        __import__(package_name)

# Check and install openai
ensure_package_installed("gradio")
ensure_package_installed("transformers")
ensure_package_installed("gtts")


# Load the image captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_description(image):
    """Generates a textual description of the given image using a pre-trained BLIP model."""
    inputs = processor(image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs)
    description = processor.decode(output[0], skip_special_tokens=True)
    return description

def text_to_speech(text):
    """Converts text to speech using gTTS and returns the audio file path."""
    tts = gTTS(text=text, lang='en')
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

def process_image(image):
    """Processes the uploaded image to generate description and return audio file."""
    description = generate_description(image)
    return description

def get_audio(description):
    """Generates the audio file for the given description."""
    return text_to_speech(description)

# Build Gradio Interface
with gradio.Blocks() as demo:
    gradio.Markdown("# Image Description and Audio Transcript App")
    gradio.Markdown("Upload an image to get an AI-generated description. Click the button to hear the description.")
    
    with gradio.Row():
        image_input = gradio.Image(type="pil")
        text_output = gradio.Textbox(label="Generated Description")
    
    generate_btn = gradio.Button("Generate Description")
    audio_btn = gradio.Button("Click here for an audio transcript")
    audio_output = gradio.Audio()
    
    generate_btn.click(process_image, inputs=[image_input], outputs=[text_output])
    audio_btn.click(get_audio, inputs=[text_output], outputs=[audio_output])
    
# Launch the Gradio app
demo.launch()
