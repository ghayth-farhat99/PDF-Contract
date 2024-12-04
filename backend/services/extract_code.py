import re


def extract_code(response: str, language_hint: str) -> str:
    match = re.search(fr"```{language_hint}\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    match = re.search(r"```.*?\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    if "contract" in response and "{" in response:  # Basic Solidity code sanity check
        return response.strip()
    
    return "Error: No valid code found."