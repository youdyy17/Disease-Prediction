from data.readCSV import load_data

# Load data once
data = load_data()

def search_symptoms(query: str):
    # Example: Replace with real search logic or database query
    symptoms = data['diseases'].tolist()  # Replace 'symptom_column' with the actual column name
    results = [item for item in symptoms if query.lower() in item.lower()]
    return results