import os


def save_unit_test_code(code):
    """
    Save the unit test code to a JavaScript file in the predefined directory.

    Args:
        code (str): The JavaScript unit test code to be saved.

    Returns:
        str: Path to the saved file.

    Raises:
        IOError: If there's an error during file writing.
    """
    base_dir = "./hardhat_test_env/test"
    file_path = os.path.join(base_dir, "test.js")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write the code to the file
    with open(file_path, "w") as file:
        file.write(code)

    return file_path


def save_smart_contract_code(code):
    """
    Save the smart contract code to a JavaScript file in the predefined directory.

    Args:
        code (str): The JavaScript smart contract code to be saved.

    Returns:
        str: Path to the saved file.

    Raises:
        IOError: If there's an error during file writing.
    """
    base_dir = "./hardhat_test_env/contracts"
    file_path = os.path.join(base_dir, "contract.sol")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Write the code to the file
    with open(file_path, "w") as file:
        file.write(code)

    return file_path
