import streamlit as st
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech
import librosa
import soundfile as sf
import os
import sounddevice as sd
from scipy.io.wavfile import write

# Load models
@st.cache_resource
def load_models():
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained("facebook/seamless-m4t-v2-large")
    return processor, model

processor, model = load_models()

# Supported languages
supported_languages = {
    "Bengali": "ben",
    "English": "eng",
    "Spanish": "es",
    "German": "deu",
    "Arabic": "arb",
    "Hindi": "hin",
    "Japanese": "jpn",
    "Korean": "kor",
    "Portuguese": "por",
}

# Initialize session state
if "audio_array" not in st.session_state:
    st.session_state.audio_array = None
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "language_code" not in st.session_state:
    st.session_state.language_code = None

# Apply CSS for dark mode
st.markdown(
    """
    <style>
        body {
            background-color: #121212; /* Dark background */
            color: #e0e0e0; /* Light text */
        }
        .block-container {
            background-color: #1e1e1e; /* Dark container */
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.5);
        }
        h1, h2, h3, h4, h5, h6 {
            color: #bb86fc; /* Accent headers */
        }
        .stButton > button {
            background-color: #bb86fc; /* Accent button */
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #7c4dff; /* Darker accent on hover */
        }
        .stSelectbox, .stSlider, .stFileUploader {
            color: white;
        }
        .stAudio {
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Description
st.title("🎮 GamerBuddy AI - Your Gaming Language Ally")

st.write(
    """
    GamerBuddy AI is your ultimate gaming partner that breaks language barriers. Whether you're 
    strategizing with teammates or coordinating epic multiplayer battles, GamerBuddy AI makes sure 
    you can communicate fluently in your fellow gamer’s language. No more language differences 
    holding you back – unite, strategize, and conquer with ease. Level up your gaming experience 
    today! 🕹️🔥
    """
)

# Audio Input Section
st.subheader("🎙️ Speak Like a Pro")
input_option = st.radio("Select How You Want to Communicate:", ["Upload Voice Commands", "Record Live Gaming Chat"], horizontal=True)

sampling_rate = 16000

if input_option == "Record Live Gaming Chat":
    st.markdown("🎤 **Record Your Live Gaming Chat**")
    duration = st.slider("Recording Duration (seconds):", 5, 60, 10)
    if st.button("🎮 Start Recording"):
        st.write("Recording... Please wait.")
        try:
            recorded_audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype="float32")
            sd.wait()
            file_name = "recorded_audio.wav"
            write(file_name, sampling_rate, (recorded_audio * 32767).astype("int16"))
            st.success("Recording complete! 🎉")
            st.audio(file_name, format="audio/wav")

            st.session_state.audio_path = file_name
            st.session_state.audio_array, _ = librosa.load(file_name, sr=sampling_rate)
        except Exception as e:
            st.error(f"Error during recording: {e}")

elif input_option == "Upload Voice Commands":
    st.markdown("📤 **Upload Your Gaming Voice Commands**")
    uploaded_file = st.file_uploader("Choose a file (wav/mp3):", type=["wav", "mp3"])
    if uploaded_file is not None:
        try:
            file_name = "uploaded_audio.wav"
            with open(file_name, "wb") as f:
                f.write(uploaded_file.read())
            st.success("File uploaded successfully! 🎉")
            st.audio(file_name, format="audio/wav")

            st.session_state.audio_path = file_name
            st.session_state.audio_array, _ = librosa.load(file_name, sr=sampling_rate)
        except Exception as e:
            st.error(f"Error loading uploaded audio: {e}")

# Language Selection Section
if st.session_state.audio_array is not None:
    st.subheader("🌐 Choose Your Gamer Buddy's Language")
    tgt_lang = st.selectbox("Select Language:", list(supported_languages.keys()))
    st.session_state.language_code = supported_languages[tgt_lang]

# Translation Section
if st.session_state.audio_array is not None and st.session_state.language_code:
    st.subheader("🎙️ Transform Your Voice to Another Language")
    if st.button("🎮 Translate"):
        with st.spinner("Translating your gaming chat..."):
            try:
                audio_inputs = processor(audios=st.session_state.audio_array, return_tensors="pt")

                output = model.generate(**audio_inputs, tgt_lang=st.session_state.language_code)[0]
                translated_audio = output.cpu().numpy().squeeze()

                output_file_path = f"translated_audio_{st.session_state.language_code}.wav"
                sf.write(output_file_path, translated_audio, samplerate=sampling_rate)

                st.success("Chat Translated! Ready to Win 🎮🔥")
                st.audio(output_file_path, format="audio/wav")

                st.markdown("📥 **Download Your Translated Voice Chat**")
                with open(output_file_path, "rb") as file:
                    st.download_button(
                        label="Download Translated Audio",
                        data=file,
                        file_name=f"translated_audio_{st.session_state.language_code}.wav",
                        mime="audio/wav",
                    )

                st.info("Your original gaming voice chat is still available for further translation! 🌟")
            except Exception as e:
                st.error(f"Error during translation: {e}")
