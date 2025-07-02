import React, { useEffect, useState } from 'react';
import axios from 'axios';

function ProcessingPage({ videoId, onDone }) {
  const [status, setStatus] = useState("processing");
  const [progress, setProgress] = useState(0);
  const [startTime, setStartTime] = useState(Date.now());

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`http://localhost:8000/status/${videoId}`);
        if (res.data.status === "done") {
          clearInterval(interval);
          onDone();
        } else {
          setStatus(res.data.status || "processing");
          // Simulate progress based on time elapsed (max 90%)
          const timeElapsed = Date.now() - startTime;
          const newProgress = Math.min(90, Math.floor((timeElapsed / 30000) * 100));
          setProgress(newProgress);
        }
      } catch (err) {
        setStatus("error");
        clearInterval(interval);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [videoId, onDone, startTime]);

  const getStatusEmoji = () => {
    switch (status) {
      case "processing":
        return "⚙️";
      case "analyzing":
        return "🔍";
      case "error":
        return "❌";
      default:
        return "⏳";
    }
  };

  const getStatusMessage = () => {
    switch (status) {
      case "processing":
        return "Processing your video";
      case "analyzing":
        return "Analyzing performance metrics";
      case "error":
        return "An error occurred";
      default:
        return "Please wait";
    }
  };

  return (
    <div style={{
      maxWidth: '600px',
      margin: '0 auto',
      padding: '40px 20px',
      textAlign: 'center'
    }}>
      <div style={{
        fontSize: '48px',
        marginBottom: '20px',
        animation: 'spin 2s linear infinite'
      }}>
        {getStatusEmoji()}
      </div>

      <h2 style={{
        color: '#333',
        marginBottom: '30px',
        fontSize: '24px'
      }}>
        {getStatusMessage()}
      </h2>

      <div style={{
        width: '100%',
        height: '8px',
        backgroundColor: '#e0e0e0',
        borderRadius: '4px',
        overflow: 'hidden',
        marginBottom: '20px'
      }}>
        <div style={{
          width: `${progress}%`,
          height: '100%',
          backgroundColor: '#2196f3',
          transition: 'width 0.5s ease',
          borderRadius: '4px',
          background: 'linear-gradient(45deg, #2196f3, #64b5f6)'
        }} />
      </div>

      <p style={{
        color: '#666',
        fontSize: '16px',
        margin: '20px 0'
      }}>
        Estimated progress: {progress}%
      </p>

      <div style={{
        marginTop: '30px',
        padding: '20px',
        backgroundColor: '#f5f5f5',
        borderRadius: '10px'
      }}>
        <h3 style={{ color: '#333', marginBottom: '15px' }}>What's happening?</h3>
        <ul style={{
          listStyle: 'none',
          padding: 0,
          margin: 0,
          textAlign: 'left'
        }}>
          <li style={{ margin: '10px 0', color: '#666' }}>
            ✓ Video upload complete
          </li>
          <li style={{ margin: '10px 0', color: '#666' }}>
            {progress > 30 ? "✓" : "○"} Analyzing facial expressions
          </li>
          <li style={{ margin: '10px 0', color: '#666' }}>
            {progress > 60 ? "✓" : "○"} Processing body language
          </li>
          <li style={{ margin: '10px 0', color: '#666' }}>
            {progress > 80 ? "✓" : "○"} Generating final report
          </li>
        </ul>
      </div>

      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
}

export default ProcessingPage;
