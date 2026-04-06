import React, { useEffect, useMemo, useState } from 'react';
import { PredictionResponse, SymptomsResponse } from '../types';
import './searchSymthom.css';

const API_BASE = 'http://127.0.0.1:5001/api';

function SearchSymthom() {
    const [allSymptoms, setAllSymptoms] = useState<string[]>([]);
    const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
    const [query, setQuery] = useState('');
    const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        const loadSymptoms = async () => {
            try {
                setError('');
                const response = await fetch(`${API_BASE}/symptoms`);
                if (!response.ok) {
                    throw new Error('Failed to load symptoms.');
                }
                const data: SymptomsResponse = await response.json();
                setAllSymptoms(data.symptoms);
            } catch (err) {
                setError((err as Error).message);
            }
        };

        loadSymptoms();
    }, []);

    const filteredSymptoms = useMemo(() => {
        const q = query.trim().toLowerCase();
        const base = q
            ? allSymptoms.filter((symptom) => symptom.toLowerCase().includes(q))
            : allSymptoms;
        return base.slice(0, 80);
    }, [allSymptoms, query]);

    const toggleSymptom = (symptom: string) => {
        setPrediction(null);
        setSelectedSymptoms((prev) =>
            prev.includes(symptom)
                ? prev.filter((item) => item !== symptom)
                : [...prev, symptom]
        );
    };

    const getPrediction = async () => {
        if (selectedSymptoms.length === 0) {
            setError('Please select at least one symptom.');
            return;
        }

        try {
            setLoading(true);
            setError('');
            const response = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symptoms: selectedSymptoms,
                    top_n: 5,
                }),
            });

            if (!response.ok) {
                throw new Error('Prediction failed. Please try again.');
            }

            const data: PredictionResponse = await response.json();
            setPrediction(data);
        } catch (err) {
            setError((err as Error).message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="prediction-page">
            <header className="hero">
                <p className="eyebrow">AI Health Assistant</p>
                <h1>Disease Prediction</h1>
                <p className="hero-subtitle">
                    Search and select symptoms to get the top 5 likely diseases with confidence scores.
                </p>
            </header>

            <div className="content-grid">
                <section className="panel">
                    <div className="panel-head">
                        <h2>Choose Symptoms</h2>
                        <span className="badge">{filteredSymptoms.length} shown</span>
                    </div>

                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Search symptoms..."
                        className="search-input"
                    />

                    <div className="symptom-list">
                        {filteredSymptoms.length === 0 && (
                            <p className="empty-state">No symptoms match your search.</p>
                        )}

                        {filteredSymptoms.map((symptom) => {
                            const checked = selectedSymptoms.includes(symptom);
                            return (
                                <label key={symptom} className={`symptom-item ${checked ? 'selected' : ''}`}>
                                    <input
                                        type="checkbox"
                                        checked={checked}
                                        onChange={() => toggleSymptom(symptom)}
                                    />
                                    <span>{symptom}</span>
                                </label>
                            );
                        })}
                    </div>
                </section>

                <section className="panel">
                    <div className="panel-head">
                        <h2>Prediction Result</h2>
                        <span className="badge">{selectedSymptoms.length} selected</span>
                    </div>

                    <button
                        onClick={getPrediction}
                        disabled={loading || selectedSymptoms.length === 0}
                        className="primary-btn"
                    >
                        {loading ? 'Predicting...' : 'Predict Top 5 Diseases'}
                    </button>

                    {selectedSymptoms.length > 0 && (
                        <div className="chip-wrap" aria-label="Selected symptoms list">
                            {selectedSymptoms.slice(0, 18).map((symptom) => (
                                <span className="chip" key={symptom}>{symptom}</span>
                            ))}
                            {selectedSymptoms.length > 18 && (
                                <span className="chip more">+{selectedSymptoms.length - 18} more</span>
                            )}
                        </div>
                    )}

                    {error && <p className="error-text">{error}</p>}

                    {!prediction && !loading && !error && (
                        <p className="empty-state">Choose symptoms and run prediction to view results.</p>
                    )}

                    {prediction && (
                        <div className="result-wrap">
                            <p className="model-meta">
                                <strong>Model:</strong> {prediction.model} • {prediction.accuracy.toFixed(2)}% validation accuracy
                            </p>

                            {prediction.predictions.map((item) => (
                                <div key={item.disease} className="result-item">
                                    <div className="result-head">
                                        <span>{item.disease}</span>
                                        <strong>{item.probability.toFixed(2)}%</strong>
                                    </div>
                                    <div className="progress-track">
                                        <div
                                            className="progress-fill"
                                            style={{ width: `${Math.max(0, Math.min(100, item.probability))}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </section>
            </div>
        </div>
    );
}

export default SearchSymthom;