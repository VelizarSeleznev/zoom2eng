from gemini import Gemini
from dotenv import load_dotenv
import os
import re
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

gemini = Gemini(GOOGLE_API_KEY)


def tokenize_text(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())


def identify_unknown_words(text: str) -> List[str]:
    prompt = f"""
    Проанализируй следующий текст и верни список слов или фраз, 
    которые являются сленгом или необычными выражениями, характерными для 'зумерского' языка:

    {text}

    Верни только список слов/фраз, разделенных запятыми, без дополнительных пояснений.
    """
    response = gemini.query(prompt)
    return [word.strip() for word in response.split(',')]


def search_word(word: str) -> List[str]:
    results = DDGS().text(f"{word} meaning slang", max_results=5)
    return [result['title'] for result in results]


def analyze_word(word: str, context: List[str]) -> str:
    prompt = f"Проанализируй слово или фразу: '{word}'. Контекст:\n"
    prompt += "\n".join(context)
    prompt += "\nПредоставь краткое определение этого выражения в контексте современного интернет-сленга:"
    return gemini.query(prompt)


def translate_text(text: str, word_definitions: Dict[str, str]) -> str:
    prompt = "Переведи следующий текст с интернет-сленга на классический английский:\n"
    prompt += text + "\n\n"
    prompt += "Используй эти определения для необычных слов и выражений:\n"
    for word, definition in word_definitions.items():
        prompt += f"{word}: {definition}\n"
    return gemini.query(prompt)


def main(text: str):
    unknown_words = identify_unknown_words(text)
    print(unknown_words)
    word_definitions = {}

    for word in unknown_words:
        search_results = search_word(word)
        definition = analyze_word(word, search_results)
        word_definitions[word] = definition
    print(word_definitions)

    translated_text = translate_text(text, word_definitions)
    return translated_text


# Пример использования
input_text = "Вайб токсичный, но чиллим, ловим флекс в потоке контента, хотя вся эта движуха – чистый кринж, не в кайф, надо срочно апнуть скилл"
result = main(input_text)
print(result)
