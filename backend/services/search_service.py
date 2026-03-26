def search_symptoms(query: str):
    # Example: Replace with real search logic or database query
    symptoms = ["fever", "cough", "headache", "sore throat"]
    results = [item for item in symptoms if query.lower() in item.lower()]
    return results