from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
import base64
import io


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
        print(f"Loading {path}")
        with open(path, encoding=encoding) as f:
            return f.read()


class BertEmbedding:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def embed(self, text: str) -> np.array:
        print(f"Embed {text}")
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


class VectorDb:

    def __init__(self, path: str):
        self.path = path
        self.embeddings: list[tuple[np.array, str]] = []
        self.bert = BertEmbedding()

    def train(self, chunks: list[str]):
        for chunk in chunks:
            embed = self.bert.embed(chunk)
            self.embeddings.append((embed, chunk))

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("embed|chunk\n")
            for embed in self.embeddings:
                f.write(self.to_base64(embed[0]))
                f.write("|")
                f.write(embed[1])
                f.write("\n")

    def load(self):
        print(f"Loading {self.path}")
        with open(self.path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            self.embeddings = []
            for row in reader:
                embed = self.from_base64(row["embed"])
                self.embeddings.append((embed, row["chunk"]))

    def to_base64(self, a: np.ndarray) -> str:
        buffer = io.BytesIO()
        np.save(buffer, a)
        buffer.seek(0)
        array_bytes = buffer.read()
        return base64.b64encode(array_bytes).decode('utf-8')

    def from_base64(self, array_base64_str: str) -> np.ndarray:
        array_bytes = base64.b64decode(array_base64_str)
        buffer = io.BytesIO(array_bytes)
        return np.load(buffer)

    def k_nearests(self, embed: np.array, k=1) -> np.array:
        embeds = [e[0] for e in self.embeddings]
        scores = self.bert.scores(embed, embeds)
        indexes = np.argsort(scores)[::-1][:3]
        return [(float(scores[i]), e[1]) for i, e in enumerate(self.embeddings) if i in indexes]

if __name__ == '__main__':
    # chunk = Chunk()
    # text = chunk.load("data/le_petit_prince.txt")
    # text = chunk.load("data/the_little_prince.txt")
    # paragraphes = chunk.chunk_paragraphes(text)
    # print(len(paragraphes))
    # print(paragraphes)
    # print(paragraphes[1])
    # print(paragraphes[2])
    # bert = BertEmbedding()
    # embed1 = bert.embed(paragraphes[1])
    # print(embed1.shape)
    # embed2 = bert.embed(paragraphes[2])
    # score = bert.score(embed1, embed2)
    # print(score)
    db = VectorDb("data/vector.db")
    # db.train(paragraphes)
    # db.save()
    db.load()
    s = "Dessines moi un mouton"
    embed = db.bert.embed(s)
    scores = db.k_nearests(embed)
    print(scores)



