export interface SymptomsResponse {
    count: number;
    symptoms: string[];
}

export interface DiseasePrediction {
    disease: string;
    probability: number;
}

export interface PredictionResponse {
    selected_symptoms: string[];
    unknown_symptoms: string[];
    predictions: DiseasePrediction[];
    model: string;
    accuracy: number;
    total_symptoms_selected: number;
}