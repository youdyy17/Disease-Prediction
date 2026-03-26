from ml_service import get_symptoms


def search_symptoms(query: str) -> list[str]:
    symptoms = get_symptoms()
    q = query.strip().lower()
    if not q:
        return symptoms[:25]
    return [item for item in symptoms if q in item.lower()][:25]