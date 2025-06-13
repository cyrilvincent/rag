from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    sentence_embedding = torch.mean(last_hidden_states, dim=1).numpy()
    return sentence_embedding

# Example texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaps over a sleepy dog.",
    "This sentence is completely different from the others."
]

# Generate embeddings for texts
embeddings = [get_sentence_embedding(text) for text in texts]

# Query text
query_text = "The quick red fox jumps over the lazy dog."
query_embedding = get_sentence_embedding(query_text)

# Compute cosine similarities
similarities = cosine_similarity(query_embedding, np.vstack(embeddings))

# Print query text
print (f"Query text: {query_text}")

# Print similarities
for i, text in enumerate(texts):
    print(f"Similarity with '{text}': {similarities[0][i]*100:.1f}%")