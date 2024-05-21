#curl -X POST \
#  https://api.cloudflare.com/client/v4/accounts/60d3f1a1e61b9e9bc3df93ae58a4fde9/ai/run/@hf/mistral/mistral-7b-instruct-v0.2 \
#  -H 'Authorization: Bearer KMvshDD2hGhNic-enwixyJ3xG-BaXh43IDcPOppO' \
#  -d '{"messages":[{"role":"system","content":"You are a friendly assistant that helps write stories"},{"role":"user","content":"Write a short story about a llama that goes on a journey to find an orange cloud "}], "stream": true}'

PROMPT="$1"
http -j --default-scheme=https \
  api.cloudflare.com/client/v4/accounts/60d3f1a1e61b9e9bc3df93ae58a4fde9/ai/run/@hf/mistral/mistral-7b-instruct-v0.2 \
  Authorization:'Bearer KMvshDD2hGhNic-enwixyJ3xG-BaXh43IDcPOppO' \
  messages:='[{"role":"system","content":"You are a friendly assistant that helps write stories"},{"role":"user","content":"Write a short story about a llama that goes on a journey to find an orange cloud "}]' \
  stream:=true