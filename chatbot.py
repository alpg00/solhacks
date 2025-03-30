import os
from dotenv import load_dotenv
import openai
# Load environment variables from the .env file
load_dotenv()

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("openai version:", openai.__version__)

# Set the OpenAI API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please check your .env file or environment variable.")

client = OpenAI(api_key=api_key)


def get_chatbot_response(question):
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
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=150,
        temperature=0.5)
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"There was an error processing your request: {e}"
    return answer
