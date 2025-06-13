import csv
import pickle
import numpy as np
import os
import tempfile
import tensorflow_hub as hub

print(os.path.join(tempfile.gettempdir(), "tfhub_modules"))

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #"https://tfhub.dev/google/universal-sentence-encoder-lite/2" #"https://tfhub.dev/google/universal-sentence-encoder/4" #"https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = hub.load(module_url)
print(f"module {module_url} loaded")

def embed(chunk: str) -> np.array:
    return model([chunk])[0].numpy()

def score(v1: np.array, v2: np.array) -> float:
    return np.inner(v1, v2)

if __name__ == '__main__':
    chunk = """Lorsque j'avais six ans j'ai vu, une fois, une magnifique image, dans un livre sur la forêt vierge qui s'appelait "Histoires Vécues". Ça représentait un serpent boa qui avalait un fauve. Voilà la copie du dessin."""
    print(chunk)
    print(len(chunk))
    vector = embed(chunk)
    print(vector.shape)
    print(vector)
    chunk2 = "Le petit prince parle d'un boa qui avait avalé un lion"
    vector2 = embed(chunk2)
    score2 = score(vector, vector2)
    print(score2)

