#!/usr/bin/env python3
import os
from typing import Callable
import requests
from requests.auth import AuthBase
import json
import argparse
import logging
import http
import dotenv


def BearerAuth(AuthBase):
    def __init__(self, api_key):
        self.api_key = api_key

    def __call__(self, r):
        r.headers["Authorization"] = f"Bearer {self.api_key}"
        return r

def process_response(response, extract_text: Callable[[str], str]):
    for line in response.iter_lines():
        if len(line) == 0: 
            continue

        if args.debug:
            if log_count <= 3:
                print(line)
                log_count += 1
            else:
                exit(0)

        if line.startswith(b'data:'):
            line = line[5:]
            try:
                msg = json.loads(line)
                print(extract_text(msg), end='', flush=True)
            except:
                pass
 
def kimi_llm_api_call(endpoint, prompt: str, **kwargs):
    conv = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            }
        ]
    }]
    use_search=True

    if "file" in kwargs and kwargs["file"] is not None and len(kwargs["file"]) > 0:
        conv[0]["content"].append({
                      "type": "file",
                      "file_url": {
                          "url": kwargs["file"]
                      }
                  })
        use_search=False

    if "image" in kwargs and kwargs["image"] is not None and len(kwargs["image"]) > 0:
        conv[0]["content"].append({
                      "type": "image_url",
                      "image_url": {
                          "url": kwargs["image"]
                      }
                  })
        use_search=False
                             
    payload = {
        'messages': conv, 'model': model, "stream": True, "use_search": use_search, "max_tokens": 10000
    }
    response = requests.post(
        endpoint,
        json=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"},
        stream=True)
    process_response(response, extract_text=lambda msg: msg['choices'][0]['delta']['content'])


def openai_llm_api_call(endpoint, prompt: str, **kwargs):
    """Call the LLM API with the given endpoint and parameters."""
    conv = [
        {"role": "user", "content": prompt},
    ]
    response = requests.post(endpoint, json={
        'messages': conv, 'model': model, "stream": True, "max_tokens": 2048
        },
        # auth=BearerAuth(api_key), # won't work, why?
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"},
        stream=True)
    process_response(response, extract_text=lambda msg: msg['choices'][0]['delta']['content'])

def gemini_llm_api_call(endpoint, prompt: str, **kwargs):
    """Call the LLM API with the given endpoint and parameters."""
    conv = [{
      "role": "user",
      "parts": [
        {
          "text": prompt
        }
      ]
    }]

    url = f"{endpoint}/{model}:streamGenerateContent"
    response = requests.post(
        url,
        params={
            "alt": "sse",
            "key": api_key
        },
        json={
            'contents': conv,
            "generationConfig": {
                "temperature": 0.9,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 2048,
                "stopSequences": []
            }
        },
        stream=True)

    process_response(response, extract_text=lambda msg: msg['candidates'][0]['content']["parts"][0]["text"])

if __name__ == '__main__':
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", '-p', required=True, type=str, help="The prompt to send to the LLM API.")
    parser.add_argument("--debug", '-d', action='store_true', help="Enable debug mode.")
    parser.add_argument("--kind", '-k', help="either use gemini or openai API", default="openai")
    parser.add_argument("--file-url", '-u', help="file url for kimi")
    parser.add_argument("--image-url", '-i', help="image url for kimi")

    args = parser.parse_args()

    if args.debug:
        http.client.HTTPConnection.debuglevel = 1
        logging.getLogger().setLevel(logging.DEBUG)

    if args.kind == "openai":
        endpoint = os.getenv("OPENAI_ENDPOINT") or 'https://gpt35.sonald.me/v1/chat/completions'
        api_key = os.getenv("OPENAI_API_KEY") or 'sk-llm'
        model = os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
        api_call = openai_llm_api_call
    elif args.kind == "kimi":
        endpoint = os.getenv("KIMI_ENDPOINT") or 'https://kimi.sonald.me/v1/chat/completions'
        api_key = os.getenv("KIMI_API_KEY") or ''
        model = os.getenv("KIMI_MODEL") or "kimi"
        api_call = kimi_llm_api_call
    else:
        endpoint = os.getenv("GOOGLE_ENDPOINT") or 'https://generativelanguage.googleapis.com/v1beta/models'
        api_key = os.getenv("GOOGLE_API_KEY") or ''
        model = os.getenv("GOOGLE_MODEL") or "gemini-1.5-pro-latest"
        api_call = gemini_llm_api_call

    api_call(endpoint, args.prompt, file=args.file_url, image=args.image_url)

