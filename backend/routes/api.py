from flask import Blueprint, jsonify, request
from ml_service import get_model_summary, get_symptoms, predict_top_diseases
from services.search_service import search_symptoms

api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@api_bp.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "")
    results = search_symptoms(query)
    return jsonify({"query": query, "results": results})


@api_bp.route("/symptoms", methods=["GET"])
def list_symptoms():
    symptoms = get_symptoms()
    return jsonify({"count": len(symptoms), "symptoms": symptoms})


@api_bp.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    symptoms = payload.get("symptoms", [])
    top_n = payload.get("top_n", 5)

    if not isinstance(symptoms, list):
        return jsonify({"error": "'symptoms' must be a list of symptom names."}), 400

    try:
        top_n = int(top_n)
    except (TypeError, ValueError):
        return jsonify({"error": "'top_n' must be an integer."}), 400

    top_n = max(1, min(top_n, 10))
    result = predict_top_diseases(symptoms, top_n)
    return jsonify(result)


@api_bp.route("/model/info", methods=["GET"])
def model_info():
    return jsonify(get_model_summary())
