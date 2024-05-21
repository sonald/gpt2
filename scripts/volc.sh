#!/bin/bash


HTTPS_OPTS="--default-scheme=https --verify no"
ENDPOINT=https://maas-api.ml-platform-cn-beijing.volces.com/api/v1
http  $HTTPS_OPTS -j $ENDPOINT/chat \
    Content-Type:"application/json" \
    AccessKeyID:"" \
    SecretAccessKey:"" \
    model="$MODEL" \
    max_tokens:=3096 \
    temperature:=0.8 \
    stream:=$STREAMING \
    messages:="[{\"role\": \"user\", \"content\": \"$PROMPT\"}]" | while IFS= read -r line; do