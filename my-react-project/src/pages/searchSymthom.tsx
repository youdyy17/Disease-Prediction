//search symthom
import React, { useState } from 'react';
function SearchSymthom() {
    const [symthom, setSymthom] = useState('');
    const [result, setResult] = useState('');
    const handleSearch = () => {        // Here you can implement your search logic, for example, making an API call to fetch results based on the symthom
        setResult(`Results for "${symthom}"`);
    };
    return (
        <div>
            <p>What are your symptoms?</p>
            <input
                type="text"
                value={symthom}
                onChange={(e) => setSymthom(e.target.value)}
                placeholder="Enter symthom..."
            />
            <button onClick={handleSearch}>Search</button>
            <p>{result}</p>
        </div>
    );
}

export default SearchSymthom;