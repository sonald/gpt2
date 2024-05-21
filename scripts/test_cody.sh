#!/usr/bin/env bash 
# parse and get args below from command line

OPTS=$(getopt -o Suhde:k:m:p:s: --long streaming,usage,help,debug,endpoint:,api-key:,model:,prompt:,style: -n "$0" -- "$@")

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

print_help() {
    echo "Usage: $0 [options] [prompt]"
    echo "Options:"
    echo "  -S, --streaming"
    echo "  -u, --usage"
    echo "  -h, --help"
    echo "  -d, --debug"
    echo "  -e, --endpoint"
    echo "  -k, --api-key <key>     Set the API key"
    echo "  -m, --model <model>     Set the model"
    echo "  -p, --prompt <prompt>   Set the prompt"
}

STREAMING=false
while true; do 
  case "$1" in
    -S ) STREAMING=true; shift ;;
    -u | --usage ) USAGE=1; shift ;;
    -h | --help ) print_help; exit 0; shift ;;
    -d | --debug ) DEBUG=1; shift ;;
    -e | --endpoint ) ENDPOINT="$2"; shift 2 ;;
    -k | --api-key ) API_KEY="$2"; shift 2 ;;
    -m | --model ) MODEL="$2"; shift 2 ;;
    -p | --prompt ) PROMPT="$2"; shift 2 ;;
    -s | --style ) STYLE="$2"; shift 2 ;;
    -- ) shift; break;;
    *) echo "Invalid option: $1"; break ;;
  esac
done

if [ -z "$PROMPT" ]; then
  PROMPT="$*"
fi


if [ -z "$PROMPT" ]; then
  PROMPT="write a bottom-up merge sort in rust"
fi

HTTPS_OPTS="--default-scheme=https --verify no"
if [ x$STYLE != x ]; then
  if [ -f scripts/style_config.sh ]; then
    source scripts/style_config.sh
  fi

  if [ x$STYLE == xollam ]; then
    HTTPS_OPTS=
  fi

  case "$STYLE" in
    "fake" | "ollama" | "infini" | "fireworks"  | "fw" | "openai" | \
     "groq" | "cf" | "gateway" | "gw" | "kimi" )
      STYLE_UP=$(echo $STYLE | tr '[:lower:] ' '[:upper:]' )

      ENDPOINT_VAR="${STYLE_UP}_ENDPOINT"
      ENDPOINT=${!ENDPOINT_VAR:-$DEFAULT_ENDPOINT}
      API_KEY_VAR="${STYLE_UP}_API_KEY"
      API_KEY=${!API_KEY_VAR:-$DEFAULT_API_KEY}
      MODEL_VAR="${STYLE_UP}_MODEL"
      MODEL=${!MODEL_VAR:-$DEFAULT_MODEL}
      ;;

    * )
      ENDPOINT=$DEFAULT_ENDPOINT
      API_KEY=$DEFAULT_API_KEY
      MODEL=$DEFAULT_MODEL
      ;;
  esac

fi


if [ x"$DEBUG" == x1 ]; then
  echo "DEBUG: ENDPOINT: $ENDPOINT \n API_KEY: $API_KEY \n MODEL: $MODEL \n PROMPT: $PROMPT \n STYLE: $STYLE"
fi

if [ x"$STREAMING" == xtrue ]; then
  http  $HTTPS_OPTS -j $ENDPOINT/chat/completions \
    Content-Type:"application/json" Authorization:"Bearer $API_KEY" \
    model="$MODEL" \
    max_tokens:=3096 \
    temperature:=0.8 \
    stream:=$STREAMING \
    messages:="[{\"role\": \"user\", \"content\": \"$PROMPT\"}]" | while IFS= read -r line; do
      if [ x"$DEBUG" == x1 ]; then
        echo "DEBUG: response: $line"
      else
        if [ x$STYLE == x"infini" ]; then 
            echo ${line} | jq -j '.choices.[0].delta.content'
        else
          if [ "${line:6}" != "[DONE]" ]; then
            echo ${line:6} | jq -j '.choices.[0].delta.content'
          fi
        fi
      fi
    done

else
  response=$(http $HTTPS_OPTS -j $ENDPOINT/chat/completions \
    Content-Type:"application/json" Authorization:"Bearer $API_KEY" \
    model="$MODEL" \
    max_tokens:=3096 \
    temperature:=0.8 \
    stream:=$STREAMING \
    messages:="[{\"role\": \"user\", \"content\": \"$PROMPT\"}]")

  if [ x"$DEBUG" == x1 ]; then
    echo "DEBUG: response: $response"
  else
    echo "$response" | jq -j '.choices.[0].message.content'  #| bat -l md
    if [ $? != 0 ]; then
      echo "$response" 
    fi
  fi

  if [ x"$USAGE" == x1 ]; then
    echo "$response" | jq -r '.usage'
  fi
fi



#curl https://api.anthropic.com/v1/messages \
#     --header "x-api-key: $ANTHROPIC_API_KEY" \
#     --header "anthropic-version: 2023-06-01" \
#     --header "content-type: application/json" \
#     --data \
#'{
#    "model": "claude-3-opus-20240229",
#    "max_tokens": 1024,
#    "messages": [
#        {"role": "user", "content": "Hello, world"}
#    ]
#}'
