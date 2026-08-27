from openai import OpenAI
import json


class LLMClient:
    def __init__(self, cfg):
        # openai调用GLM-4.6V
        self.client = OpenAI(
            api_key=cfg.api_key,  # apiKey
            base_url=cfg.url,
        )
        self.model = cfg.model  # 选用模型

    def chat(self, messages, tools=None, temperature=0.2) -> dict:
        """发送历史 + 工具定义，返回一条 assistant 消息。
        返回格式统一为: {"content": str|None, "tool_calls": [ToolCall]}
        """
        resp = self.client.chat.completions.create(
            model=self.model,  # glm-4.6v
            messages=messages,
            tools=tools,  # None 时不下发该字段
            temperature=temperature,
        )
        msg = resp.choices[0].message
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,  # JSON 字符串
                    },
                }
                for tc in (msg.tool_calls or [])
            ],
        }
