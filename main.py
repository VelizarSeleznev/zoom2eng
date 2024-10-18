from gemini import Gemini
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

gemini = Gemini(GOOGLE_API_KEY)
chat_response = gemini.query("Write a short story about a robot learning to paint.")
print(chat_response)