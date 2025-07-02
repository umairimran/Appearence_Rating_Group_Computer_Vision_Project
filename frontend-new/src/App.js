import React, { useState } from 'react';
import UploadVideo from './UploadVideo';
import ProcessingPage from './ProcessingPage';
import ResultsPage from './ResultsPage';

function App() {
  const [stage, setStage] = useState("upload"); // "upload" → "processing" → "results"
  const [videoId, setVideoId] = useState(null); // Filename from backend

  const handleUploaded = (id) => {
    setVideoId(id);      // Save uploaded filename
    setStage("processing");
  };

  const handleProcessingDone = () => {
    setStage("results");
  };

  return (
    <div style={{ padding: '30px' }}>
      <h1>🎥 AI Video Analyzer</h1>
      {stage === "upload" && <UploadVideo onUploaded={handleUploaded} />}
      {stage === "processing" && <ProcessingPage videoId={videoId} onDone={handleProcessingDone} />}
      {stage === "results" && <ResultsPage videoId={videoId} />}
    </div>
  );
}

export default App;
