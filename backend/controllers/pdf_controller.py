import os
import zipfile
from flask import Blueprint, request, jsonify, send_file, abort
from services.pdf_helper_services import PDFHelperServices
from services.analyze_contract_services import (
    compile_contract,
    use_slither,
    use_hardhat,
)
from services.analyze_test_services import use_eslint
from config import Config

pdf_bp = Blueprint("pdf", __name__)
# Initialize PDFHelperServices (assuming it's available globally)
ollama_api_base_url = Config.OLLAMA_API_BASE_URL
model_name = Config.MODEL

pdf_helper = PDFHelperServices(
    ollama_api_base_url=ollama_api_base_url, model_name=model_name
)


@pdf_bp.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return 'No pdf file provided', 400
    
    pdf_file = request.files['file']
    
    # Path to save the PDF as 'contract.pdf'
    pdf_folder = 'pdf'
    os.makedirs(pdf_folder, exist_ok=True)  # Ensure the directory exists
    pdf_file_path = os.path.join(pdf_folder, 'contract.pdf')
    
    # Delete old 'contract.pdf' if it exists
    if os.path.exists(pdf_file_path):
        os.remove(pdf_file_path)
    
    # Save the new PDF file with the name 'contract.pdf'
    pdf_file.save(pdf_file_path)

    # Ask the PDFHelperServices for relevant information
    response= pdf_helper.ask_for_Extract_Information(pdf_file_path)
    #response = pdf_helper.ask(
    #    pdf_file_path=pdf_file_path,
    #    question="You are a legal expert. I am a blockchain developer seeking to extract relevant information from the provided legal contract PDF. Please identify all key clauses and provisions necessary for building a smart contract.Requirements: Focus on clarity and relevance.Do not include any code or technical details."
    #)
    return jsonify({'response': response})


@pdf_bp.route("/generate_unit_test", methods=["POST"])
def generate_unit_test():
    if not request.json or 'extract_informations' not in request.json:
        return jsonify({'error': 'No extract_informations provided'}), 400
    
    extract_informations = request.json['extract_informations']
    pdf_helper.update_information(extract_informations)
    # Generate unit tests based on the provided information
    unit_tests_response = pdf_helper.ask_for_generate_unit_test(information=extract_informations)
    
    return jsonify({'unit_tests': unit_tests_response})


""" @pdf_bp.route('/generate_prompt', methods=['POST'])
def generate_prompt():
    if not request.json or 'unit_tests' not in request.json:
        return jsonify({'error': 'No unit tests provided'}), 400
    
    prompt = request.json['unit_tests']
    pdf_helper.update_UnitTests(prompt)
    # Generate a prompt based on the unit tests
    prompt_response = pdf_helper.ask_for_generate_prompt(context=prompt)
    
    return jsonify({'prompt_response': prompt_response}) """


@pdf_bp.route("/generate_smart_contract", methods=["POST"])
def generate_smart_contract():

    # prompt = request.json['messagePrompt']
    prompt = request.json["unit_tests"]
    pdf_helper.update_UnitTests(prompt)
    # Generate a smart contract based on the provided prompt
    smart_contract_response = pdf_helper.ask_for_generate_smart_contract()

    return jsonify({"smart_contract": smart_contract_response})


@pdf_bp.route("/analyze_contract", methods=["POST"])
def analyze_contract():

    (
        compilation_result_returncode,
        compilation_result_stdout,
        compilation_result_stderr,
    ) = compile_contract()
    compilation_step = {
        "status": "success" if compilation_result_returncode == 0 else "failed",
        "stdout": (
            compilation_result_stdout if compilation_result_returncode == 0 else "N/A"
        ),
        "stderr": (
            "N/A" if compilation_result_returncode == 0 else compilation_result_stderr
        ),
    }

    if compilation_result_returncode:
        response = {"compilation_step": compilation_step}
        return jsonify(response), 200

    slither_result_returncode, slither_result_stdout, slither_result_stderr = (
        use_slither()
    )
    slither_step = {
        "status": "success" if slither_result_returncode == 0 else "failed",
        "stdout": slither_result_stdout if slither_result_returncode == 0 else "N/A",
        "stderr": "N/A" if slither_result_returncode == 0 else slither_result_stderr,
    }

    hardhat_result_returncode, hardhat_result_stdout, hardhat_result_stderr = (
        use_hardhat()
    )
    hardhat_step = {
        "status": "success" if hardhat_result_returncode == 0 else "failed",
        "stdout": hardhat_result_stdout,
        "stderr": hardhat_result_stderr,
    }

    # Combine results into a single dictionary
    response = {
        "compilation_step": compilation_step,
        "slither_step": slither_step,
        "hardhat_step": hardhat_step,
    }

    return jsonify(response), 200


@pdf_bp.route("/analyze_contract_solc", methods=["POST"])
def analyze_contract_solc():

    (
        compilation_result_returncode,
        compilation_result_stdout,
        compilation_result_stderr,
    ) = compile_contract()
    solc_response = {
        "status": "success" if compilation_result_returncode == 0 else "failed",
        "stdout": (
            compilation_result_stdout if compilation_result_returncode == 0 else "N/A"
        ),
        "stderr": (
            "N/A" if compilation_result_returncode == 0 else compilation_result_stderr
        ),
    }

    response = {"compilation_step": solc_response}
    return jsonify(response), 200


@pdf_bp.route("/analyze_contract_slither", methods=["POST"])
def analyze_contract_slither():

    slither_result_returncode, slither_result_stdout, slither_result_stderr = (
        use_slither()
    )
    slither_response = {
        "status": "success" if slither_result_returncode == 0 else "failed",
        "stdout": slither_result_stdout if slither_result_returncode == 0 else "N/A",
        "stderr": "N/A" if slither_result_returncode == 0 else slither_result_stderr,
    }
    response = {"slither_response": slither_response}
    return jsonify(response), 200


@pdf_bp.route("/analyze_contract_hardhat", methods=["POST"])
def analyze_contract_hardhat():
    hardhat_result_returncode, hardhat_result_stdout, hardhat_result_stderr = (
        use_hardhat()
    )
    hardhat_response = {
        "status": "success" if hardhat_result_returncode == 0 else "failed",
        "stdout": hardhat_result_stdout,
        "stderr": hardhat_result_stderr,
    }
    
    response = {"hardhat_response": hardhat_response}
    return jsonify(response), 200


@pdf_bp.route("/download")
def download_file():
    folder_path = "./hardhat_test_env"
    zip_path = "./hardhat_test_env.zip"

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        return abort(404, description="Resource not found")

    # Create a zip file excluding node_modules
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            # Exclude node_modules directory
            if "node_modules" in dirs:
                dirs.remove("node_modules")
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

    # Check if the zip file was created
    if not os.path.isfile(zip_path):
        return abort(500, description="Error creating zip file")

    # Send the zip file
    return send_file(zip_path, as_attachment=True)


# To configure the Llama model
@pdf_bp.route("/configure_model", methods=["POST"])
def configure_model():
    if not request.json or "temperature" not in request.json:
        return jsonify({"error": "No temperature provided"}), 400

    # Extract the temperature from the request
    temperature = request.json["temperature"]
    # Validate the temperature range (e.g., between 0 and 1)
    if not (0 <= temperature <= 1):
        return jsonify({"error": "Temperature must be between 0 and 1"}), 400

    pdf_helper.set_model_temperature(temperature)

    return jsonify({"message": f"Model temperature set to {temperature}"}), 200

@pdf_bp.route('/analyze_test', methods=['GET'])
def analyze_test():
    eslint_result_returncode, eslint_result_stdout, eslint_result_stderr = use_eslint()
    eslint_step = {
        'status': 'success' if eslint_result_returncode == 0 else 'failed',
        'stdout': eslint_result_stdout,
        'stderr': eslint_result_stderr
    }
    return jsonify(eslint_step), 200

@pdf_bp.route('/regenerate_unit_test_eslint', methods=['POST'])
def regenerate_unit_test():
    eslint_results = request.json["eslint_results"]
    print("eslint_results : ", eslint_results)
    unit_tests = pdf_helper.ask_for_regenerate_unit_test_eslint(eslint_results)
    print("unit_tests : ", unit_tests)
    pdf_helper.update_UnitTests(unit_tests)
    return jsonify({"unit_tests": unit_tests})

@pdf_bp.route('/regenerate_smart_contract_solc', methods=['POST'])
def regenerate_smart_contractt():
    solc_results = request.json["solc_results"]
    print("solc_results : ", solc_results)
    smart_contract = pdf_helper.ask_for_regenerate_smart_contract_solc(solc_results)
    print("smart_contract : ", smart_contract)
    return jsonify({"smart_contract": smart_contract})

@pdf_bp.route('/regenerate_smart_contract_slither', methods=['POST'])
def regenerate_smart_contractt():
    slither_results = request.json["slither_results"]
    print("slither_results : ", slither_results)
    smart_contract = pdf_helper.ask_for_regenerate_smart_contract_slither(slither_results)
    print("smart_contract : ", smart_contract)
    return jsonify({"smart_contract": smart_contract})

@pdf_bp.route('/regenerate_smart_contract_hardhat', methods=['POST'])
def regenerate_smart_contractt():
    hardhat_results = request.json["hardhat_results"]
    print("hardhat_results : ", hardhat_results)
    smart_contract,unit_test = pdf_helper.ask_for_correct_smart_contract_unit_test_hardhat(hardhat_results)
    print("smart_contract : ", smart_contract)
    print("unit_test : ", unit_test)
    return jsonify({"smart_contract": smart_contract, "unit_test": unit_test})