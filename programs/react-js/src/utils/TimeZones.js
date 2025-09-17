import React, { useState, useEffect } from 'react';

const Timezones = () => {
  const [options, setOptions] = useState([]); // State to hold dropdown options
  const [selected, setSelected] = useState(''); // State for selected value
  const [loading, setLoading] = useState(true); // State for loading indicator

  useEffect(() => {
    // Fetch options from the API
    const fetchOptions = async () => {
      try {
        const response = await fetch('https://timeapi.io/api/timezone/availabletimezones'); // Example API
        const data = await response.json();
        const formattedOptions = data.map((user) => ({
          id: user,
          name: user,
        }));
        setOptions(formattedOptions);
        setLoading(false); // Stop loading when data is loaded
      } catch (error) {
        console.error('Error fetching options:', error);
        setLoading(false);
      }
    };

    fetchOptions();
  }, []); // Empty dependency array ensures this runs once on component mount

  const handleSelectChange = (event) => {
    setSelected(event.target.value);
  };

  return (
    <div>
      <label htmlFor="dropdown">Choose an option:</label>
      {loading ? (
        <p>Loading options...</p>
      ) : (
        <select id="dropdown" value={selected} onChange={handleSelectChange}>
          <option value="">--Select an option--</option>
          {options.map((option) => (
            <option key={option.id} value={option.id}>
              {option.name}
            </option>
          ))}
        </select>
      )}
      {selected && <p>Selected ID: {selected}</p>}
    </div>
  );
};

export default Timezones;
