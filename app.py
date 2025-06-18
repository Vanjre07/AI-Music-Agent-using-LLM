from audiocraft.models import MusicGen, AudioGen
import streamlit as st 
import torch 
import torchaudio
import os 
import numpy as np
import base64

# Load model with caching, selecting based on type and variant
@st.cache_resource
def load_model(model_type: str, model_variant: str):
    if model_type == "MusicGen":
        model = MusicGen.get_pretrained(model_variant)
    elif model_type == "AudioGen":
        model = AudioGen.get_pretrained(model_variant)
    else:
        raise ValueError("Invalid model type")
    return model

# Function to process audio: normalization and fade in/out for smoothness
def process_audio(audio_tensor, sample_rate=32000, fade_in=0.1, fade_out=0.1):
    # Normalize audio to avoid clipping and improve dynamic range
    max_val = torch.abs(audio_tensor).max()
    if max_val > 0:
        audio_tensor = audio_tensor / max_val

    # Apply fade in and fade out
    fade_in_samples = int(fade_in * sample_rate)
    fade_out_samples = int(fade_out * sample_rate)
    fade_in_curve = torch.linspace(0, 1, fade_in_samples)
    fade_out_curve = torch.linspace(1, 0, fade_out_samples)
    
    audio_tensor[..., :fade_in_samples] *= fade_in_curve
    audio_tensor[..., -fade_out_samples:] *= fade_out_curve
    return audio_tensor

# Generate audio tensor using the selected model with advanced parameters
def generate_audio_tensors(description: str, duration: int, top_k: int,
                           temperature: float, num_samples: int, model_type: str,
                           model_variant: str):
    model = load_model(model_type, model_variant)
    
    # Set generation parameters (common parameters for both models)
    model.set_generation_params(
        use_sampling=True,
        top_k=top_k,
        temperature=temperature,
        duration=duration
    )

    # Generate outputs; note that some models might return a tuple so extract the audio tensor
    outputs = model.generate(
        descriptions=[description] * num_samples, 
        progress=True,
        return_tokens=True
    )
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return outputs

def save_audio(samples: torch.Tensor, apply_processing=True):
    sample_rate = 32000
    save_path = "audio_output/"
    os.makedirs(save_path, exist_ok=True)

    if isinstance(samples, tuple):
        samples = samples[0]

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]
    
    saved_files = []
    for idx, audio in enumerate(samples):
        if apply_processing:
            audio = process_audio(audio, sample_rate)
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)
        saved_files.append(audio_path)
    return saved_files

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_icon="🎵",
    page_title="Text To Music Generator"
)

# Main app
def main():
    st.title("🎶 Text To Music Generator")
    
    with st.expander("ℹ️ About this app"):
        st.write(
            "This AI-powered app uses Meta's Audiocraft framework to generate Music from text. "
            
        )
    
    # Input for text prompt
    text_prompt = st.text_area("🎤 Enter your prompt (description)...")
    
    # Optional Instrumental prompt (for music)
    instrumental_mode = st.checkbox("🎹 Enhance with instrumental prompt")
    instrumental_prompt = ""
    if instrumental_mode:
        instrumental_prompt = st.text_area("🎼 Enter instrumental prompt (instruments, style, etc.)")
    
    # Duration slider for audio generation
    duration = st.slider("⏱️ Select duration (in seconds)", 1, 60, 30)

    # Choose the type of generation
    model_type = st.selectbox("Choose Audio Model Type", options=["MusicGen", "AudioGen"])
    # Depending on type, allow a model variant selection
    if model_type == "MusicGen":
        model_variant = st.selectbox("Choose Model Variant", options=["facebook/musicgen-small", "facebook/musicgen-large"])
    else:
        model_variant = st.selectbox("Choose Model Variant", options=["facebook/audiogen-medium"])
    
    # Advanced generation parameters
    with st.expander("🔧 Advanced Options"):
        top_k = st.slider("Top‑k (sampling parameter)", min_value=50, max_value=300, value=250)
        temperature = st.slider("Temperature (sampling diversity)", min_value=0.5, max_value=1.5, value=1.0)
        num_samples = st.number_input("Number of audio samples", min_value=1, max_value=5, value=1, step=1)
    
    if st.button("🎧 Generate Audio"):
        if not text_prompt.strip():
            st.warning("Please enter a valid prompt.")
        else:
            with st.spinner("Generating audio, please wait..."):
                try:
                    # Combine main and instrumental prompts if applicable
                    if instrumental_mode and instrumental_prompt.strip():
                        combined_prompt = f"{text_prompt.strip()} Instrumental details: {instrumental_prompt.strip()}"
                    else:
                        combined_prompt = text_prompt.strip()
                    
                    # Generate the audio tensors using the selected model
                    audio_tensors = generate_audio_tensors(
                        description=combined_prompt,
                        duration=duration,
                        top_k=top_k,
                        temperature=temperature,
                        num_samples=num_samples,
                        model_type=model_type,
                        model_variant=model_variant
                    )
                    
                    # Save and process the generated audio
                    saved_files = save_audio(audio_tensors, apply_processing=True)
                    
                    # Display all generated audio files with players and download links
                    for file in saved_files:
                        with open(file, 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                        st.audio(audio_bytes)
                        st.markdown(get_binary_file_downloader_html(file, 'Generated Audio'), unsafe_allow_html=True)
                    
                    st.success("✅ Audio generated successfully!")
                except Exception as e:
                    st.error(f"🚨 An error occurred while generating audio: {e}")

if __name__ == "__main__":
    main()
