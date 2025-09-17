import React, { useState, useEffect } from 'react';

const Clock = ({ timeZone="America/New_York" ,size = 100, color = 'grey', text = ''}) => {
  const [time, setTime] = useState(new Date());
  const styles2={ textAlign: 'center', fontSize: '24px', margin: '20px' }
  const styles = {
    width: `${size}px`,
    height: `${size}px`,
    backgroundColor: color,
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    fontSize: '16px',
    fontWeight: 'bold',
  };
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone,
    hour12: false,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  useEffect(() => {
    const timerId = setInterval(() => {
      setTime(new Date());
    }, 1000);

    // Cleanup function to clear the interval
    return () => clearInterval(timerId);
  }, []);

  return (
    <div style={styles}>
      
      <p>{time.toLocaleTimeString()}</p>
    </div>
  );
};

export default Clock;
