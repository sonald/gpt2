#!/bin/bash
#

if [ $# -ge 1 ]; then
    prompt="$@"
else
    prompt='Write a story about a magic backpack.'
fi

content=$(cat <<EOF
[ { "parts":[{ "text": "$prompt" }] } ]
EOF
)


response=$(http -j --default-scheme=https \
    generativelanguage.googleapis.com:/v1beta/models/gemini-pro:generateContent \
    key=="$API_KEY" \
    contents:="$content"
)


if echo $response | grep -q 'error'; then
    echo $response | jq '.error.message'
else
    echo "$response" | jq -r -c '.candidates[0].content.parts[0].text'
fi

exit 0


