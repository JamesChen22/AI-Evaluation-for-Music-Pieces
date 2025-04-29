import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_yamnet_embedding(audio_path, desired_sr=16000):
    """
    Load an audio file, resample it to the desired sampling rate,
    extract an embedding using YAMNet, and return the audio as well.
    """
    # Load the audio file with librosa and resample to desired_sr
    y, sr = librosa.load(audio_path, sr=desired_sr)
    
    # Run the audio through YAMNet. The model returns scores, embeddings, and spectrogram.
    scores, embeddings, spectrogram = yamnet_model(y)
    
    # Average the embeddings across time frames to obtain a single feature vector.
    embedding = tf.reduce_mean(embeddings, axis=0)
    return embedding.numpy(), y, sr

# Replace with your actual audio file path
sample_audio = 'test.mp3'
embedding, y, sr = extract_yamnet_embedding(sample_audio)

# Display YAMNet embedding information
print("Extracted embedding shape:", embedding.shape)
print("First 10 embedding features:", embedding[:10])
print("Embedding summary:")
print("  Min:", np.min(embedding))
print("  Max:", np.max(embedding))
print("  Mean:", np.mean(embedding))
print("  Std Dev:", np.std(embedding))

# Plot the YAMNet embedding
plt.figure(figsize=(12, 6))
plt.plot(embedding, marker='o', linestyle='-', markersize=3)
plt.title("YAMNet Audio Embedding")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.grid(True)
plt.show()

# -------------------------
# Now, extract a rhythm feature using librosa beat tracking:
# -------------------------

# Compute tempo and beat frames using librosa
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print("Estimated tempo (BPM):", tempo)

# Compute the onset strength envelope (useful for visualizing rhythm)
o_env = librosa.onset.onset_strength(y=y, sr=sr)
times = librosa.times_like(o_env, sr=sr)

# Plot the onset envelope with detected beats
plt.figure(figsize=(12, 4))
plt.plot(times, o_env, label="Onset Strength")
# Convert beat frames to time for plotting
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
plt.vlines(beat_times, 0, o_env.max(), color='r', alpha=0.75, linestyle='--', label='Beats')
plt.xlabel("Time (s)")
plt.ylabel("Onset Strength")
plt.title("Onset Strength and Detected Beats (Rhythm Feature)")
plt.legend()
plt.show()
