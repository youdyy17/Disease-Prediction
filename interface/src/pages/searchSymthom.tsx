import React, { useEffect, useMemo, useState } from 'react';
import { PredictionResponse, SymptomsResponse } from '../types';

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
        <div style={{ maxWidth: 900, margin: '20px auto', fontFamily: 'Arial, sans-serif' }}>
            <h2>Disease Prediction</h2>
            <p>Select symptoms, then get top 5 possible diseases.</p>

            <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search symptoms..."
                style={{ width: '100%', padding: 10, marginBottom: 12 }}
            />

            <div style={{ display: 'flex', gap: 20 }}>
                <div
                    style={{
                        flex: 1,
                        border: '1px solid #ddd',
                        borderRadius: 8,
                        padding: 12,
                        maxHeight: 340,
                        overflowY: 'auto',
                    }}
                >
                    {filteredSymptoms.map((symptom) => {
                        const checked = selectedSymptoms.includes(symptom);
                        return (
                            <label
                                key={symptom}
                                style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}
                            >
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

                <div style={{ flex: 1 }}>
                    <p><strong>Selected symptoms:</strong> {selectedSymptoms.length}</p>
                    <button
                        onClick={getPrediction}
                        disabled={loading || selectedSymptoms.length === 0}
                        style={{ padding: '10px 16px', cursor: 'pointer' }}
                    >
                        {loading ? 'Predicting...' : 'Predict Top 5 Diseases'}
                    </button>

                    {error && <p style={{ color: 'crimson', marginTop: 12 }}>{error}</p>}

                    {prediction && (
                        <div style={{ marginTop: 16 }}>
                            <p>
                                <strong>Model:</strong> {prediction.model} ({prediction.accuracy.toFixed(2)}% validation accuracy)
                            </p>
                            {prediction.predictions.map((item) => (
                                <div key={item.disease} style={{ marginBottom: 10 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <span>{item.disease}</span>
                                        <span>{item.probability.toFixed(2)}%</span>
                                    </div>
                                    <div style={{ background: '#eee', borderRadius: 8, height: 10 }}>
                                        <div
                                            style={{
                                                width: `${item.probability}%`,
                                                background: '#2f80ed',
                                                height: '100%',
                                                borderRadius: 8,
                                            }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default SearchSymthom;