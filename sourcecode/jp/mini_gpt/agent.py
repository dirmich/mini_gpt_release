"""ReActパターンのエージェントループ"""
import re
from mini_gpt.tools import TOOL_SCHEMAS, format_tools_for_prompt, AVAILABLE_FUNCTIONS


REACT_SYSTEM_PROMPT = """You are an agent that solves tasks step by step.
{tools}

Use exactly this format, one step at a time:
Thought: <your reasoning about what to do next>
Action: <tool_name>[<arguments as a simple string>]
Observation: <this will be filled in for you>

When you have enough information, respond with:
Thought: <final reasoning>
Final Answer: <your answer to the user>
"""


def parse_action(text):
    """'Action: calculator[847 * 293]' のような形式をパース"""
    match = re.search(r"Action:\s*(\w+)\[(.*?)\]", text)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def run_agent(llm_generate_fn, question, tool_functions, max_steps=5):
    """
    llm_generate_fn: (prompt: str) -> str 形式のモデル呼び出し関数
    tool_functions: {"tool_name": actual_execution_function} 辞書
    """
    tools_text = format_tools_for_prompt(TOOL_SCHEMAS)
    transcript = REACT_SYSTEM_PROMPT.format(tools=tools_text)
    transcript += f"\nQuestion: {question}\n"

    for step in range(max_steps):
        step_output = llm_generate_fn(transcript)
        transcript += step_output

        if "Final Answer:" in step_output:
            final_answer = step_output.split("Final Answer:")[-1].strip()
            return final_answer, transcript

        tool_name, arg_str = parse_action(step_output)
        if tool_name and tool_name in tool_functions:
            # 非常に単純化された引数マッピング：実戦ではツールのスキーマに合わせて精緻にパースする必要がある
            try:
                if tool_name == "calculator":
                    observation = tool_functions[tool_name](expression=arg_str)
                elif tool_name == "get_weather":
                    observation = tool_functions[tool_name](city=arg_str)
                else:
                    observation = "未知のツール形式"
            except Exception as e:
                observation = f"ツール実行エラー: {e}"
        else:
            observation = "アクションをパースできませんでした。形式を再度確認してください。"

        transcript += f"\nObservation: {observation}\n"

    return "最大ステップ数に達しましたが、最終的な回答を得られませんでした。", transcript
