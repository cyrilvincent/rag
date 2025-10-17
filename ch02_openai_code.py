from openai import OpenAI
import os

with open("data/secrets/key.secret") as f:
    os.environ["OPENAI_API_KEY"] = f.read()

with open("ch02_openai_service.py") as f:
    text = f.read()

client = OpenAI()

completion = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Ajoute des commentaires docstrings dans ce code Python"},
        {"role": "user", "content": text}
    ]
)

res = completion.choices[0].message
print(res.content)
