import os
from dotenv import load_dotenv
import openai
import base64

# Load environment variables from the .env file
load_dotenv()

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("openai version:", openai.__version__)
print("🔁 Using model: gpt-4-turbo")

# Set the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please check your .env file or environment variable.")

client = OpenAI(api_key=api_key)

def get_chatbot_response(question, image_bytes=None, image_name=None):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant specialized in discussing and explaining the graphs in a loan approval dashboard. "
                "Only answer questions related to these graphs. If the question is off-topic, reply that you can only discuss the graphs."
            )
        },
        {"role": "user", "content": question}
    ]

    if image_bytes:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}"
            }
        }
        messages.append({"role": "user", "content": [
            {"type": "text", "text": f"The user uploaded an image ({image_name}). Please interpret or use it for context if needed."},
            image_data
        ]})

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=300,
            temperature=0.5
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"There was an error processing your request: {e}"
    return answer
