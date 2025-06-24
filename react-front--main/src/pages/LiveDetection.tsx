import { useState } from 'react';

const LiveDetection = () => {
  const [isStreaming, setIsStreaming] = useState(false);

  const startStreaming = async () => {
    setIsStreaming(true);
  };

  const stopStreaming = async () => {
    await fetch('http://127.0.0.1:3000/stop-stream', {
      method: 'POST',
    });
    setIsStreaming(false);
  };

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      flexDirection: 'column',
      gap: '20px'
    }}>
      {/* Bold heading */}
      <h2 style={{
        fontSize: '2rem',
        fontWeight: 'bold',
        color: '#333',
        textAlign: 'center',
      }}>
        Live Detection
      </h2>

      {/* Streaming video or instruction */}
      {isStreaming ? (
        <img
          src="http://127.0.0.1:3000/start-stream"
          alt="Live Stream"
          style={{
            width: '640px',
            height: '480px',
            borderRadius: '10px',
            border: '2px solid #ccc',
            objectFit: 'cover'
          }}
        />
      ) : (
        <p style={{
          fontSize: '1.2rem',
          color: '#777',
          textAlign: 'center',
        }}>
          Click "Start" to activate the camera.
        </p>
      )}

      {/* Start and Stop buttons */}
      <div style={{ display: 'flex', gap: '15px' }}>
        {!isStreaming ? (
          <button onClick={startStreaming} style={buttonStyle}>
            Start Camera
          </button>
        ) : (
          <button onClick={stopStreaming} style={{ ...buttonStyle, backgroundColor: '#ff4d4f' }}>
            Stop Camera
          </button>
        )}
      </div>
    </div>
  );
};

// Button styling
const buttonStyle = {
  padding: '10px 20px',
  fontSize: '1rem',
  backgroundColor: '#4CAF50',
  color: '#fff',
  border: 'none',
  borderRadius: '8px',
  cursor: 'pointer',
  transition: 'background-color 0.3s ease',
  boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
};

export default LiveDetection;
