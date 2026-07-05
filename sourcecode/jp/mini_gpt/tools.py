"""ツールの定義、呼び出しのパース、実行を担当するモジュール"""
import json
import re


TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "指定された都市の現在の天気情報を返す",
        "parameters": {
            "city": {"type": "string", "description": "都市名 (英語)"},
        },
    },
    {
        "name": "calculator",
        "description": "四則演算の数式を計算する",
        "parameters": {
            "expression": {"type": "string", "description": "例: '847 * 293'"},
        },
    },
]


def format_tools_for_prompt(tool_schemas):
    """ツールリストをプロンプトに含めやすいテキスト形式に変換"""
    lines = ["Available tools:"]
    for tool in tool_schemas:
        params = ", ".join(
            f'{name}: {info["type"]}' for name, info in tool["parameters"].items()
        )
        lines.append(f'- {tool["name"]}({params}): {tool["description"]}')
    return "\n".join(lines)


def parse_tool_call(model_output):
    """モデルの出力から '{"tool": ..., "arguments": {...}}' 形式のJSONを探してパース"""
    match = re.search(r"\{.*\}", model_output, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


# 実際に実行可能なPython関数 (実戦では本物のAPI呼び出しに置き換える)
def get_weather(city):
    fake_weather_db = {"seoul": "18°C, Cloudy", "tokyo": "21°C, Sunny", "new york": "15°C, Rainy"}
    return fake_weather_db.get(city.lower(), f"{city}の天気情報が見つかりません")


def calculator(expression):
    # 実戦では eval の代わりに安全な数式パーサーを使用すべき (ここでは教育用に簡略化)
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "許可されていない文字が含まれている数式です"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"計算エラー: {e}"


AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "calculator": calculator,
}


def execute_tool_call(tool_call):
    """パースされたツール呼び出しを実際に実行"""
    name = tool_call.get("tool")
    arguments = tool_call.get("arguments", {})
    if name not in AVAILABLE_FUNCTIONS:
        return f"未知のツール: {name}"
    return AVAILABLE_FUNCTIONS[name](**arguments)
