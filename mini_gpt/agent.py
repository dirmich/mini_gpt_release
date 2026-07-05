"""ReAct 패턴의 에이전트 루프"""
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
    """'Action: calculator[847 * 293]' 같은 형태를 파싱"""
    match = re.search(r"Action:\s*(\w+)\[(.*?)\]", text)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def run_agent(llm_generate_fn, question, tool_functions, max_steps=5):
    """
    llm_generate_fn: (prompt: str) -> str  형태의 모델 호출 함수
    tool_functions: {"도구이름": 실제_실행_함수} 딕셔너리
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
            # 아주 단순화된 인자 매핑: 실전에서는 도구 스키마에 맞춰 정교하게 파싱해야 함
            try:
                if tool_name == "calculator":
                    observation = tool_functions[tool_name](expression=arg_str)
                elif tool_name == "get_weather":
                    observation = tool_functions[tool_name](city=arg_str)
                else:
                    observation = "알 수 없는 도구 형식"
            except Exception as e:
                observation = f"도구 실행 오류: {e}"
        else:
            observation = "행동을 파싱할 수 없었습니다. 형식을 다시 확인하세요."

        transcript += f"\nObservation: {observation}\n"

    return "최대 스텝 수에 도달했지만 최종 답변을 얻지 못했습니다.", transcript
