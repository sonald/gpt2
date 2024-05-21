import os
import dotenv
import argparse


def request_v2(prompt: str):
    from volcengine.maas import MaasService, MaasException, ChatRole
    def test_stream_chat(maas, req):
        try:
            resps = maas.stream_chat(req)
            for resp in resps:
                print(resp.choice.message.content, end='')
        except MaasException as e:
            print(e)
        print('')

    maas = MaasService('maas-api.ml-platform-cn-beijing.volces.com', 'cn-beijing')

    # set ak&sk
    # maas.set_ak(VOLC_ACCESSKEY)
    # maas.set_sk(VOLC_SECRETKEY)
    model1 = {
        "name": "skylark2-pro-4k",
        "version": "1.2", # use default version if not specified.
    }

    model2 = {
        "name": "moonshot-v1-32k",
    }

    # document: "https://www.volcengine.com/docs/82379/1099475"
    req = {
        "model": model1,
        "parameters": {
            "max_new_tokens": 1000,  # 输出文本的最大tokens限制
            "min_new_tokens": 1,  # 输出文本的最小tokens限制
            "temperature": 0.7,  # 用于控制生成文本的随机性和创造性，Temperature值越大随机性越大，取值范围0~1
            "top_p": 0.9,  # 用于控制输出tokens的多样性，TopP值越大输出的tokens类型越丰富，取值范围0~1
            "top_k": 0,  # 选择预测值最大的k个token进行采样，取值范围0-1000，0表示不生效
            "max_prompt_tokens": 32768,  # 最大输入 token 数，如果给出的 prompt 的 token 长度超过此限制，取最后 max_prompt_tokens 个 token 输入模型。
        },
        "messages": [
            {
                "role": ChatRole.USER,
                "content": prompt
            },
        ]
    }

    test_stream_chat(maas, req)

def request_v3_api_key():
    import volcenginesdkcore
    import volcenginesdkark
    from pprint import pprint
    from volcenginesdkcore.rest import ApiException

    configuration = volcenginesdkcore.Configuration()
    configuration.ak = os.environ.get('VOLC_ACCESSKEY')
    configuration.sk = os.environ.get('VOLC_SECRETKEY')
    configuration.region = "cn-beijing"
    # set default configuration
    volcenginesdkcore.Configuration.set_default(configuration)

    # ep = 'maas-api.ml-platform-cn-beijing.volces.com'
    ep = 'ep-20240515085022-g7pn9'
    # use global default configuration
    api_instance = volcenginesdkark.ARKApi()
    get_api_key_request = volcenginesdkark.GetApiKeyRequest(
        duration_seconds=30*24*3600,
        resource_type="endpoint",
        resource_ids=[ep],
    )
    
    try:
        resp = api_instance.get_api_key(get_api_key_request)
        pprint(resp)
    except ApiException as e:
        print("Exception when calling api: %s\n" % e)


def request_v3(prompt: str):
    from volcenginesdkarkruntime import Ark
    ep = 'ep-20240515085022-g7pn9'
    client = Ark()
    stream = client.chat.completions.create(
        model=ep,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue

        print(chunk.choices[0].delta.content, end="")



if __name__ == '__main__':
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", '-p', required=True, type=str)

    args = parser.parse_args()
    # request_v3_api_key()
    request_v3(args.prompt)
