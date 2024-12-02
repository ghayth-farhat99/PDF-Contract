import os
import subprocess
from flask import Flask
from flask_cors import CORS
from config import Config
from services.pdf_helper_services import PDFHelperServices, load_embedding_model
from controllers.pdf_controller import pdf_bp  # Import the blueprint
from controllers.update_files_controller import unit_test_bp

from services.pull_model import pull_model

pull_model()

# Load the embedding model
load_embedding_model(model_name=Config.EMBEDDING_MODEL_NAME)
model_name = Config.MODEL
ollama_api_base_url = Config.OLLAMA_API_BASE_URL

# Initialize PDFHelperServices
pdf_helper = PDFHelperServices(
    ollama_api_base_url=ollama_api_base_url, model_name=model_name
)

# install solidity compiler 0.8.25 and use it
subprocess.run(["solc-select", "install", "0.8.25"], capture_output=True, text=True)
subprocess.run(["solc-select", "use", "0.8.25"], capture_output=True, text=True)


if os.path.exists("hardhat_test_env/contracts") == False:
    os.mkdir("hardhat_test_env/contracts")
if os.path.exists("hardhat_test_env/test") == False:
    os.mkdir("hardhat_test_env/test")

# Create the Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Register the blueprint
app.register_blueprint(pdf_bp)
app.register_blueprint(unit_test_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
