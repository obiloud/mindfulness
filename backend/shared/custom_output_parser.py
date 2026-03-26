import re
from typing import Dict, Any


def parse_conversation_string(conversation_string: str) -> Dict[str, Any]:
    """
    Parse a string of the format "key=value" into a dictionary.

    Handles both quoted strings and numeric values.

    Args:
        conversation_string: String in format "key=value" or "key=\"value\""

    Returns:
        Dictionary with parsed key-value pairs
    """
    if not conversation_string or not isinstance(conversation_string, str):
        return {}

    # Remove leading/trailing whitespace
    conversation_string = conversation_string.strip()

    # Split by first "=" to get key and rest
    parts = conversation_string.split("=", 1)
    if len(parts) != 2:
        return {}

    key = parts[0].strip()
    value_part = parts[1].strip()

    # Extract the actual value by removing quotes if present
    # Handle both quoted and unquoted values
    value = value_part

    # Remove surrounding quotes if present
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    elif value.startswith("'") and value.endswith("'"):
        value = value[1:-1]

    # Handle numeric values (like info_score=1.0)
    try:
        # Try to parse as float first
        value = float(value)
    except ValueError:
        # If not a number, keep as string
        pass

    # Handle special cases like "summary=" with quoted content
    # We need to parse the value properly to handle nested quotes
    # Use regex to extract values properly

    # More robust parsing using regex
    # This handles cases where values contain quotes or other special characters
    pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^"\n]+)'
    matches = re.findall(pattern, conversation_string)

    if not matches:
        return {}

    # Create dictionary from matches
    result = {}
    for key_val in matches:
        key = key_val[0].strip()
        value = key_val[1].strip()

        # Handle quotes in value
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]

        # Try to parse as float
        try:
            value = float(value)
        except ValueError:
            # If not a number, keep as string
            pass

        result[key] = value

    return result

# Enhanced version that can handle multiple key-value pairs in one string


def parse_conversation_string_advanced(conversation_string: str) -> Dict[str, Any]:
    """
    Advanced parser that handles multiple key-value pairs in a single string.

    Args:
        conversation_string: String containing multiple key-value pairs separated by spaces or newlines

    Returns:
        Dictionary with parsed key-value pairs
    """
    if not conversation_string or not isinstance(conversation_string, str):
        return {}

    # Remove leading/trailing whitespace
    conversation_string = conversation_string.strip()

    # Split by spaces or newlines to handle multiple key-value pairs
    # This handles cases like "summary=value info_score=1.0"
    pairs = []
    # Split by spaces and filter out empty strings
    tokens = conversation_string.split()

    # Group tokens into key-value pairs
    i = 0
    while i < len(tokens):
        if i < len(tokens) and tokens[i].strip().endswith("="):
            # This is a malformed pair - skip
            i += 1
            continue

        if i < len(tokens) and tokens[i].strip().startswith("="):
            # This is a malformed pair - skip
            i += 1
            continue

        # Look for key-value pairs
        key = tokens[i].strip()
        if i + 1 < len(tokens):
            value_part = tokens[i + 1].strip()
            i += 2

            # Extract value by removing quotes if present
            value = value_part
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            # Try to parse as float
            try:
                value = float(value)
            except ValueError:
                pass

            pairs.append((key, value))
        else:
            i += 1

    # Create dictionary from pairs
    result = {}
    for key, value in pairs:
        result[key] = value

    return result

# Integration with your existing code


def parse_evaluation_output(evaluation_string: str) -> Dict[str, Any]:
    """
    Parse the evaluation string into a structured dictionary.

    This is specifically designed for the format:
    "summary=\"The conversation begins with a polite greeting...\" info_score=1.0"
    """
    # Use the advanced parser
    parsed = parse_conversation_string_advanced(evaluation_string)

    # Ensure required fields are present
    result = {
        "summary": parsed.get("summary", "No summary provided"),
        "info_score": parsed.get("info_score", 0.0)
    }

    # Validate that info_score is a float
    if not isinstance(result["info_score"], (int, float)):
        result["info_score"] = 0.0

    return result


# Example usage
if __name__ == "__main__":
    test_string = "summary=\"The conversation begins with a polite greeting and a check-in on emotional state. The user responds with a neutral mood ('I am fine') and reciprocates with a question about the agent's well-being. The exchange is brief, coherent, and centered on emotional connection and mutual awareness. There is no drift in topic, and the latest agent message does not ask follow-up questions. The conversation provides sufficient context about the user's current state (neutral, stable) and expresses a clear intent to engage in a mindful exchange. It is short and meaningful, meeting all criteria for a mature state.\" info_score=1.0"

    parsed_result = parse_evaluation_output(test_string)
    print("Parsed result:", parsed_result)
