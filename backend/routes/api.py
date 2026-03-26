from flask import Blueprint, jsonify, request
from backend.services.search_service import search_symptoms

api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


@api_bp.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "")
    # TODO: Implement search logic
    results = search_symptoms(query)
    return jsonify({"query": query, "results": results})
