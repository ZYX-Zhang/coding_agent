from zai import ZhipuAiClient
import json

with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

client = ZhipuAiClient(api_key=cfg['llm']['api_key'])

response = client.chat.completions.create(
    model="glm-4.6",
    messages=[
        {"role": "user", "content": "作为一名营销专家，请为我的产品创作一个吸引人的口号"},
        {"role": "assistant", "content": "当然，要创作一个吸引人的口号，请告诉我一些关于您产品的信息"},
        {"role": "user", "content": "智谱开放平台"}
    ],
    thinking={
        "type": "enabled",  # 启用深度思考模式
    },
    max_tokens=10000,  # 最大输出 tokens
    temperature=1.0  # 控制输出的随机性
)

# 获取完整回复
print(response.choices[0].message)
