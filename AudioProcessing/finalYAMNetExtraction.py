import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import csv
import urllib.request

# -------------------------------
# Define the list of features to be output
# -------------------------------

# Numerical features to output (will be computed or set by heuristics)
numerical_features = [
    "time_signature", "speechiness", "danceability", "duration_ms",
    "energy", "mode", "instrumentalness", "valence", "key", "tempo",
    "loudness", "acousticness", "liveness"
]
# Categorical features to output (set to default values)
categorical_features = ["playlist_subgenre", "playlist_genre", "type"]

# -------------------------------
# Load the pre-trained YAMNet model from TensorFlow Hub
# -------------------------------
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)
print("YAMNet model loaded.")

# -------------------------------
# Function to load the YAMNet class map using a new URL
# -------------------------------
def load_yamnet_class_map():
    """
    Loads the YAMNet class mapping from Google Cloud Storage.
    This mapping is used to identify indices for classes (e.g. "speech").
    """
    # New reliable URL for the YAMNet class map hosted on Google Cloud Storage
    url = "https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv"
    class_map = []
    with urllib.request.urlopen(url) as response:
        lines = [line.decode('utf-8') for line in response.readlines()]
        reader = csv.reader(lines)
        next(reader)  # Skip header line
        for row in reader:
            # CSV columns: index, mid, display_name. We use display_name.
            class_map.append(row[2].strip().lower())
    return class_map

# Load the YAMNet class map and determine the index for "speech"
yamnet_class_map = load_yamnet_class_map()
if "speech" in yamnet_class_map:
    speech_index = yamnet_class_map.index("speech")
else:
    speech_index = None
print("YAMNet class map loaded. Speech index:", speech_index)

# -------------------------------
# Feature Extraction Function
# -------------------------------
def extract_features_from_mp3(file_path):
    """
    Extracts audio features from the provided .mp3 file.
    Features include:
      - time_signature: defaulted to 4 (common time)
      - speechiness: estimated using YAMNet's "speech" output
      - danceability: heuristic based on tempo and energy
      - duration_ms: track duration in milliseconds
      - energy: average RMS energy
      - mode: default "major"
      - instrumentalness: 1 - speechiness (heuristic)
      - valence: default neutral value 0.5
      - key: estimated using chroma features
      - tempo: estimated BPM from beat tracking
      - loudness: average dB of the audio
      - acousticness: default placeholder 0.5
      - liveness: default placeholder 0.5
    Also outputs categorical features set as defaults.
    """
    # Load the audio for musical features extraction at 22050 Hz
    y, sr = librosa.load(file_path, sr=22050)
    
    # Duration in milliseconds
    duration_ms = librosa.get_duration(y=y, sr=sr) * 1000

    # Estimate tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Compute RMS energy and average it
    rms = librosa.feature.rms(y=y)
    energy = np.mean(rms)
    
    # Compute loudness: average decibel value from the STFT magnitude
    S = np.abs(librosa.stft(y))
    loudness = np.mean(librosa.amplitude_to_db(S, ref=np.max))
    
    # Estimate key using chroma features, then pick the highest component
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    key_idx = np.argmax(chroma)
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = key_names[key_idx]
    
    # Default mode is set as "major"
    mode = "major"
    
    # Default time signature, common time
    time_signature = 4
    
    # -------------------------------
    # Use YAMNet to estimate "speechiness"
    # -------------------------------
    # YAMNet expects audio at 16000 Hz.
    y_yamnet, _ = librosa.load(file_path, sr=16000)
    waveform = tf.convert_to_tensor(y_yamnet, dtype=tf.float32)
    waveform = tf.reshape(waveform, [1, -1])
    scores, _, _ = yamnet_model(waveform)
    
    if speech_index is not None:
        speechiness = float(tf.reduce_mean(scores[:, speech_index]).numpy())
    else:
        speechiness = 0.0
        
    # Heuristic: instrumentalness is the inverse of speechiness.
    instrumentalness = 1.0 - speechiness
    
    # Heuristic for danceability: combination of tempo and energy (scaled arbitrarily)
    danceability = min((tempo / 200) * energy, 1.0)
    
    # Placeholders for features that require more complex analysis:
    valence = 0.5       # Neutral
    acousticness = 0.5  # Default placeholder
    liveness = 0.5      # Default placeholder
    
    # Categorical features (defaults)
    playlist_subgenre = "unknown"
    playlist_genre = "unknown"
    audio_type = "mp3"  # We are processing an mp3 file
    
    # Combine all features into a dictionary
    features = {
        "time_signature": time_signature,
        "speechiness": speechiness,
        "danceability": danceability,
        "duration_ms": duration_ms,
        "energy": energy,
        "mode": mode,
        "instrumentalness": instrumentalness,
        "valence": valence,
        "key": key,
        "tempo": tempo,
        "loudness": loudness,
        "acousticness": acousticness,
        "liveness": liveness,
        "playlist_subgenre": playlist_subgenre,
        "playlist_genre": playlist_genre,
        "type": audio_type
    }
    
    return features

# -------------------------------
# Main Execution: Process test.mp3 and output features
# -------------------------------
input_file = "test.mp3"

if not os.path.exists(input_file):
    print(f"File {input_file} does not exist. Please ensure it's in the working directory.")
else:
    features = extract_features_from_mp3(input_file)
    print("Extracted Features from test.mp3:")
    for feature_name, value in features.items():
        print(f"{feature_name}: {value}")
