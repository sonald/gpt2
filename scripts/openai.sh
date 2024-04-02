#!/usr/bin/env bash 
# parse and get args below from command line

OPTS=$(getopt -o Suhde:k:m:p:s: --long streaming,usage,help,debug,endpoint:,api-key:,model:,prompt:,style: -n "$0" -- "$@")

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

print_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -d, --debug             Enable debug mode"
  echo "  -S                     Enable streaming output"
  echo "  -e, --endpoint <url>    Set the API endpoint"
  echo "  -k, --api-key <key>     Set the API key"
  echo "  -m, --model <model>     Set the model"
  echo "  -p, --prompt <prompt>   Set the prompt"
  echo "  -s, --style <style>     Set the style: fireworks(fw), ehco, openai, ohmygpt, groq, cf, gateway(gw), ollama"
  echo "  -h, --help              print this help"
  echo "  -u, --usage             print token usage"
  echo "  --                      End of options"
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

if [ x$STYLE != x ]; then
  if [ -f scripts/style_config.sh ]; then
    source scripts/style_config.sh
  fi

  HTTPS_OPTS="--default-scheme=https --verify no"
  case "$STYLE" in
    "ollama" )
      HTTPS_OPTS=
      ENDPOINT=$OLLAMA_ENDPOINT
      API_KEY=$OLLAMA_API_KEY
      MODEL=$OLLAMA_MODEL
      ;;


    "infini" )
      ENDPOINT=$INFINI_ENDPOINT
      API_KEY=$INFINI_API_KEY
      MODEL=$INFINI_MODEL
      ;;

    "fireworks"  | "fw" )
      ENDPOINT=$FW_ENDPOINT
      API_KEY=$FW_API_KEY
      MODEL=$FW_MODEL
      ;;

    "ehco" )
      ENDPOINT=$EHCO_ENDPOINT
      API_KEY=$EHCO_API_KEY
      MODEL=$EHCO_MODEL
      ;;

    "openai" )
      ENDPOINT=$OPENAI_ENDPOINT
      API_KEY=$OPENAI_API_KEY
      MODEL=$OPENAI_MODEL
      ;;

    "ohmygpt" )
      ENDPOINT=$OHMYGPT_ENDPOINT
      API_KEY=$OHMYGPT_API_KEY
      MODEL=$OHMYGPT_MODEL
      ;;

    "groq" )
      ENDPOINT=$GROQ_ENDPOINT
      API_KEY=$GROQ_API_KEY
      MODEL=$GROQ_MODEL
      ;;

    "cf" )
      ENDPOINT=$CF_ENDPOINT
      API_KEY=$CF_API_KEY
      MODEL=$CF_MODEL
      ;;

    "gateway" | "gw" )
      ENDPOINT=$GW_ENDPOINT
      API_KEY=$GW_API_KEY
      MODEL=$GW_MODEL
      ;;

    * )
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
        if [ $STYLE == "infini" ]; then 
            echo ${line} | jq -j '.choices.[0].delta.content'
        else
          if [ "${line:6}" != "[DONE]" ]; then
            echo ${line:6} | jq -j '.choices.[0].delta.content'
          fi
        fi
      fi
    done

else
  response=$(http  $HTTPS_OPTS -j $ENDPOINT/chat/completions \
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
