import os
from mistralai import Mistral

with open("data/secrets/mistral.secret") as f:
    os.environ["MISTRAL_API_KEY"] = f.read()
api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)
print(chat_response.choices[0].message.content)

# Roles
# A user message is a message sent from a human to the chatbot. This is typically how we first interact with any chatbot.
# A system message is an optional message that sets the context for a chatbot, such as modifying its personality or providing specific instructions. For example, we often ask the chatbot to act in a specific role, such as “You are an expert in marketing and sales from a Fortune 500 company”.
# An assistant message is a message sent by the chatbot back to the user. It is meant to reply to a previous user message by following its instructions.
# A tool message only appears in the context of function calling, it is used at the final response formulation step when the model has to format the tool call’s output for the user. Learn more about Mistral’s function calling.