"""Module responsible for tool definition, call parsing, and execution"""
import json
import re


TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Returns the current weather information for a given city",
        "parameters": {
            "city": {"type": "string", "description": "City name (English)"},
        },
    },
    {
        "name": "calculator",
        "description": "Calculates arithmetic expressions",
        "parameters": {
            "expression": {"type": "string", "description": "e.g., '847 * 293'"},
        },
    },
]


def format_tools_for_prompt(tool_schemas):
    """Converts the tool list into text suitable for inclusion in a prompt"""
    lines = ["Available tools:"]
    for tool in tool_schemas:
        params = ", ".join(
            f'{name}: {info["type"]}' for name, info in tool["parameters"].items()
        )
        lines.append(f'- {tool["name"]}({params}): {tool["description"]}')
    return "\n".join(lines)


def parse_tool_call(model_output):
    """Finds and parses JSON in the form of '{\"tool\": ..., \"arguments\": {...}}' from model output"""
    match = re.search(r"\{.*\}", model_output, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


# Actual executable Python functions (In production, replace with real API calls)
def get_weather(city):
    fake_weather_db = {"seoul": "18°C, Cloudy", "tokyo": "21°C, Sunny", "new york": "15°C, Rainy"}
    return fake_weather_db.get(city.lower(), f"Weather information for {city} could not be found")


def calculator(expression):
    # In production, a safe expression parser should be used instead of eval (Simplified here for educational purposes)
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Expression contains disallowed characters"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Calculation error: {e}"


AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "calculator": calculator,
}


def execute_tool_call(tool_call):
    """Actually executes the parsed tool call"""
    name = tool_call.get("tool")
    arguments = tool_call.get("arguments", {})
    if name not in AVAILABLE_FUNCTIONS:
        return f"Unknown tool: {name}"
    return AVAILABLE_FUNCTIONS[name](**arguments)
