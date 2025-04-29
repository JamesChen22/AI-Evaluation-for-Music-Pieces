import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained YAMNet model from TensorFlow Hub
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_yamnet_embedding(audio_path, desired_sr=16000):
    """
    Load an audio file, resample it to the desired sampling rate, and extract an embedding using YAMNet.
    """
    # Load the audio file with librosa and resample to desired_sr
    y, sr = librosa.load(audio_path, sr=desired_sr)
    
    # Run the audio through YAMNet. The model returns scores, embeddings, and spectrogram.
    scores, embeddings, spectrogram = yamnet_model(y)
    
    # Average the embeddings across time frames to obtain a single feature vector.
    embedding = tf.reduce_mean(embeddings, axis=0)
    return embedding.numpy()

# Test the function with a sample audio file (update the path to your file)
sample_audio = 'test.mp3'
embedding = extract_yamnet_embedding(sample_audio)

# Print embedding shape and summary statistics
print("Extracted embedding shape:", embedding.shape)
print("First 10 features:", embedding[:10])
print("Summary statistics:")
print("  Min:", np.min(embedding))
print("  Max:", np.max(embedding))
print("  Mean:", np.mean(embedding))
print("  Std Dev:", np.std(embedding))

# Plot the embedding vector for visual inspection
plt.figure(figsize=(12, 6))
plt.plot(embedding, marker='o', linestyle='-', markersize=3)
plt.title("YAMNet Audio Embedding")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.grid(True)
plt.show()
