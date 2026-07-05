"""Agent loop for the ReAct pattern"""
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
    """Parses forms like 'Action: calculator[847 * 293]'"""
    match = re.search(r"Action:\s*(\w+)\[(.*?)\]", text)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def run_agent(llm_generate_fn, question, tool_functions, max_steps=5):
    """
    Model call function in the form of llm_generate_fn: (prompt: str) -> str
    tool_functions: A dictionary of {"tool_name": actual_execution_function}
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
            # Very simplified argument mapping: In practice, this must be parsed precisely according to the tool schema
            try:
                if tool_name == "calculator":
                    observation = tool_functions[tool_name](expression=arg_str)
                elif tool_name == "get_weather":
                    observation = tool_functions[tool_name](city=arg_str)
                else:
                    observation = "Unknown tool format"
            except Exception as e:
                observation = f"Tool execution error: {e}"
        else:
            observation = "Could not parse action. Please check the format."

        transcript += f"\nObservation: {observation}\n"

    return "Reached maximum steps without obtaining a final answer.", transcript
