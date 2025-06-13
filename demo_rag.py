from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Chunk:

    def __init__(self, chunk_size=128, max_chunk_size=1024):
        self.chunk_size = chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_paragraphes(self, text: str) -> list[str]:
        lines = text.split("\n")
        paragraphes = []
        paragraphe = ""
        for line in lines:
            if line.strip() == "" and len(paragraphe) > self.chunk_size:
                if paragraphe.strip() != "":
                    if len(paragraphe) > self.max_chunk_size:
                        paragraphes.append(paragraphe[:self.max_chunk_size].strip())
                        paragraphe = line
                    else:
                        paragraphes.append(paragraphe.strip())
                        paragraphe = ""
            else:
                paragraphe += line.strip() + " "
        return paragraphes

    def load(self, path: str, encoding="utf-8") -> str:
        with open(path, encoding=encoding) as f:
            return f.read()

class BertEmbeding:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def embed(self, text: str) -> np.array:
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        sentence_embedding = torch.mean(last_hidden_states, dim=1).numpy()
        return sentence_embedding

    def score(self, embed1: np.array, embed2: np.array) -> float:
        score = cosine_similarity(embed1, np.vstack([embed2]))
        return score[0][0]

    def scores(self, embed1: np.array, embeds: list[np.array]) -> np.array:
        score = cosine_similarity(embed1, np.vstack(embeds))
        return score[0]


if __name__ == '__main__':
    chunk = Chunk()
    text = chunk.load("data/le_petit_prince.txt")
    # text = chunk.load("data/the_little_prince.txt")
    paragraphes = chunk.chunk_paragraphes(text)
    print(len(paragraphes))
    print(paragraphes)
    print(paragraphes[1])
    print(paragraphes[2])
    bert = BertEmbeding()
    embed1 = bert.embed(paragraphes[1])
    print(embed1.shape)
    embed2 = bert.embed(paragraphes[2])
    score = bert.score(embed1, embed2)
    print(score)
