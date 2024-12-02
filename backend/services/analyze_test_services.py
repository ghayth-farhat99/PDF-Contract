import subprocess

def use_eslint():
    # Run eslint tests
    result = subprocess.run(["npx", "eslint", "./hardhat_test_env/test/test.js"], capture_output=True, text=True, shell=True)
    eslint_result_returncode = result.returncode
    eslint_result_stdout = str(result.stdout)
    eslint_result_stderr = result.stderr
    return(eslint_result_returncode, eslint_result_stdout, eslint_result_stderr)