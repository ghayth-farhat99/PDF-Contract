from flask import Blueprint, request, jsonify, abort
from services.update_files_services import save_unit_test_code, save_smart_contract_code

unit_test_bp = Blueprint("unit_test", __name__)


@unit_test_bp.route("/save_unit_test", methods=["POST"])
def save_unit_test():
    """
    Endpoint to save unit test code to a specified location.
    """
    data = request.get_json()

    # Validate input data
    if not data or "code" not in data:
        abort(400, description="Invalid input. 'code' field is required.")

    unit_test_code = data["code"]

    try:
        # Use the function to save the unit test
        file_path = save_unit_test_code(unit_test_code)
        return (
            jsonify(
                {"message": "Unit test saved successfully", "file_path": file_path}
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@unit_test_bp.route("/save_smart_contract", methods=["POST"])
def save_smart_contract():
    """
    Endpoint to save smart contract code to a specified location.
    """
    data = request.get_json()

    # Validate input data
    if not data or "code" not in data:
        abort(400, description="Invalid input. 'code' field is required.")

    smart_contract_code = data["code"]

    try:
        # Use the function to save the unit test
        file_path = save_smart_contract_code(smart_contract_code)
        return (
            jsonify(
                {"message": "Smart contract saved successfully", "file_path": file_path}
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
