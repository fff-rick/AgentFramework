import asyncio
import json

from app.core.intent.recognizer import IntentRecognizer


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    async def chat(self, messages, model_preference=None, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "model_preference": model_preference,
                "kwargs": kwargs,
            }
        )
        return type("Response", (), {"content": self.content})()


def test_recognize_uses_multi_turn_history_and_model_json() -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "intent": "文档",
                "sub_intent": "总结",
                "confidence": 0.91,
                "slots": {"document": "上一个文件"},
                "rationale": "用户追问此前上传文件",
                "needs_clarification": False,
                "clarification_question": "",
            },
            ensure_ascii=False,
        )
    )
    recognizer = IntentRecognizer(llm)

    result = asyncio.run(
        recognizer.recognize(
            messages=[
                {"role": "user", "content": "我刚上传了一份季度报告"},
                {"role": "assistant", "content": "文件已收到"},
                {"role": "user", "content": "帮我提炼重点"},
            ]
        )
    )

    assert result.intent == "文档"
    assert result.sub_intent == "总结"
    assert result.slots == {"document": "上一个文件"}
    sent_conversation = json.loads(llm.calls[0]["messages"][1]["content"])["conversation"]
    assert [message["content"] for message in sent_conversation] == [
        "我刚上传了一份季度报告",
        "文件已收到",
        "帮我提炼重点",
    ]
    assert llm.calls[0]["kwargs"] == {"temperature": 0, "max_tokens": 300}


def test_recognize_never_falls_back_to_keyword_rules() -> None:
    llm = FakeLLM("not json")
    recognizer = IntentRecognizer(llm)

    result = asyncio.run(recognizer.recognize(query="帮我搜索最新资料"))

    assert result.intent == "未知"
    assert result.sub_intent == "澄清"
    assert result.needs_clarification is True
    assert result.confidence == 0


def test_low_confidence_model_result_requests_clarification() -> None:
    llm = FakeLLM(
        '{"intent":"问答","sub_intent":"知识问答","confidence":0.2,'
        '"slots":{},"rationale":"信息不足","needs_clarification":false,'
        '"clarification_question":"您希望了解哪个具体方面？"}'
    )
    recognizer = IntentRecognizer(llm, confidence_threshold=0.55)

    result = asyncio.run(recognizer.recognize(query="那个怎么做"))

    assert result.needs_clarification is True
    assert asyncio.run(recognizer.clarify("那个怎么做", result)) == "您希望了解哪个具体方面？"
