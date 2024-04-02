import requests
from requests.auth import AuthBase
import json
import argparse
import logging
import http


endpoint = 'https://deepseek-api.sonald.me/v1/chat/completions'
api_key = 'sk-llm'
model="openai"

def BearerAuth(AuthBase):
    def __init__(self, api_key):
        self.api_key = api_key

    def __call__(self, r):
        r.headers["Authorization"] = f"Bearer {self.api_key}"
        return r


def llm_api_call(endpoint, prompt: str):
    """Call the LLM API with the given endpoint and parameters."""
    global log_count
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

    # print(response.text)
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
                if 'content' in msg['choices'][0]['delta']:
                    print(msg['choices'][0]['delta']['content'], end='', flush=True)
                # else:
                #     print(msg['choices'][0]['delta'])
            except:
                # line == '[DONE]'
                pass

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", '-p', type=str, help="The prompt to send to the LLM API.")
parser.add_argument("--debug", '-d', action='store_true', help="Enable debug mode.")

args = parser.parse_args()
log_count = 0

if args.debug:
    http.client.HTTPConnection.debuglevel = 1
    logging.getLogger().setLevel(logging.DEBUG)

if args.prompt:
    llm_api_call(endpoint, args.prompt)
else:
    llm_api_call(endpoint, "What is the meaning of life?")
