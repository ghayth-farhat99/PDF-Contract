import os
import subprocess

def compile_contract():
    # Compilation step
    compilation_result = subprocess.run(
        ["solc", "--gas", "--bin", "./hardhat_test_env/contracts/contract.sol"],
        capture_output=True,
        text=True,
    )
    compilation_result_returncode = compilation_result.returncode
    compilation_result_stdout = str(compilation_result.stdout)
    compilation_result_stderr = compilation_result.stderr
    return (
        compilation_result_returncode,
        compilation_result_stdout,
        compilation_result_stderr,
    )


def use_slither():
    # Slither step
    slither_result = subprocess.run(
        ["slither", "./hardhat_test_env/contracts"], capture_output=True, text=True
    )
    slither_result_returncode = slither_result.returncode
    slither_result_stdout = str(slither_result.stdout)
    slither_result_stderr = slither_result.stderr
    return (slither_result_returncode, slither_result_stdout, slither_result_stderr)

def use_hardhat():
    try:
        result = subprocess.run(
            ["npx", "hardhat", "test"],
            cwd="./hardhat_test_env",
            capture_output=True,
            text=True,
            shell=True
        )
        hardhat_result_returncode = result.returncode
        hardhat_result_stdout = str(result.stdout)
        hardhat_result_stderr = result.stderr
        return hardhat_result_returncode, hardhat_result_stdout, hardhat_result_stderr

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running Hardhat: {e}")
        return None, None, None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None, None
