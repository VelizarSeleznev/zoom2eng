import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold


class Gemini:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

    def query(self, prompt, **kwargs):
        generation_config = self._create_generation_config(kwargs)
        safety_settings = self._create_safety_settings(kwargs)

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            # safety_settings=safety_settings
        )

        return response.text

    def chat(self, messages, **kwargs):
        generation_config = self._create_generation_config(kwargs)
        safety_settings = self._create_safety_settings(kwargs)

        chat = self.model.start_chat(history=[])
        response = chat.send_message(
            messages,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        return response.text

    def _create_generation_config(self, kwargs):
        return GenerationConfig(
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 1.0),
            top_k=kwargs.get('top_k', 1),
            max_output_tokens=kwargs.get('max_output_tokens', 8192),
            stop_sequences=kwargs.get('stop_sequences', None),
            candidate_count=kwargs.get('candidate_count', 1)
        )

    def _create_safety_settings(self, kwargs):
        return {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        }
