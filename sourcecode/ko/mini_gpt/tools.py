"""도구 정의, 호출 파싱, 실행을 담당하는 모듈"""
import json
import re


TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "주어진 도시의 현재 날씨 정보를 반환한다",
        "parameters": {
            "city": {"type": "string", "description": "도시 이름 (영문)"},
        },
    },
    {
        "name": "calculator",
        "description": "사칙연산 수식을 계산한다",
        "parameters": {
            "expression": {"type": "string", "description": "예: '847 * 293'"},
        },
    },
]


def format_tools_for_prompt(tool_schemas):
    """도구 목록을 프롬프트에 넣기 좋은 텍스트로 변환"""
    lines = ["Available tools:"]
    for tool in tool_schemas:
        params = ", ".join(
            f'{name}: {info["type"]}' for name, info in tool["parameters"].items()
        )
        lines.append(f'- {tool["name"]}({params}): {tool["description"]}')
    return "\n".join(lines)


def parse_tool_call(model_output):
    """모델 출력에서 '{"tool": ..., "arguments": {...}}' 형태의 JSON을 찾아 파싱"""
    match = re.search(r"\{.*\}", model_output, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


# 실제 실행 가능한 파이썬 함수들 (실전에서는 진짜 API 호출로 교체)
def get_weather(city):
    fake_weather_db = {"seoul": "18°C, 흐림", "tokyo": "21°C, 맑음", "new york": "15°C, 비"}
    return fake_weather_db.get(city.lower(), f"{city}의 날씨 정보를 찾을 수 없습니다")


def calculator(expression):
    # 실전에서는 eval 대신 안전한 수식 파서를 사용해야 함 (여기서는 교육용으로 단순화)
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "허용되지 않은 문자가 포함된 수식입니다"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"계산 오류: {e}"


AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "calculator": calculator,
}


def execute_tool_call(tool_call):
    """파싱된 도구 호출을 실제로 실행"""
    name = tool_call.get("tool")
    arguments = tool_call.get("arguments", {})
    if name not in AVAILABLE_FUNCTIONS:
        return f"알 수 없는 도구: {name}"
    return AVAILABLE_FUNCTIONS[name](**arguments)
