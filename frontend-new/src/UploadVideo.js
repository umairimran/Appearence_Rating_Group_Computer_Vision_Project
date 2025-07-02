import React, { useState, useRef } from 'react';
import axios from 'axios';

function UploadVideo({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith('video/')) {
      setFile(droppedFile);
    } else {
      alert('Please drop a video file!');
    }
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a video file first!");

    try {
      setUploading(true);
      const formData = new FormData();
      formData.append("file", file);

      const res = await axios.post("http://localhost:8000/upload/", formData, {
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        }
      });
      
      onUploaded(res.data.filename);
    } catch (error) {
      alert('Upload failed: ' + error.message);
      setUploading(false);
    }
  };

  return (
    <div className="upload-container" style={{
      maxWidth: '600px',
      margin: '0 auto',
      padding: '20px',
      textAlign: 'center'
    }}>
      <div 
        className="drop-zone"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `3px dashed ${isDragging ? '#2196f3' : '#ccc'}`,
          borderRadius: '10px',
          padding: '40px 20px',
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: isDragging ? 'rgba(33, 150, 243, 0.1)' : '#f8f9fa',
          transition: 'all 0.3s ease',
          marginBottom: '20px'
        }}
      >
        <div style={{ marginBottom: '15px' }}>
          <span style={{ fontSize: '48px', marginBottom: '10px' }}>📹</span>
        </div>
        <h3 style={{ margin: '0 0 10px 0', color: '#333' }}>
          {file ? file.name : 'Drop your video here'}
        </h3>
        <p style={{ margin: '0', color: '#666' }}>
          {file ? 'Click to change video' : 'or click to select a file'}
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
        />
      </div>

      {file && !uploading && (
        <button
          onClick={handleUpload}
          style={{
            backgroundColor: '#2196f3',
            color: 'white',
            border: 'none',
            padding: '12px 30px',
            borderRadius: '25px',
            fontSize: '16px',
            cursor: 'pointer',
            transition: 'background-color 0.3s ease',
            boxShadow: '0 2px 5px rgba(0,0,0,0.2)'
          }}
          onMouseOver={e => e.target.style.backgroundColor = '#1976d2'}
          onMouseOut={e => e.target.style.backgroundColor = '#2196f3'}
        >
          Start Analysis
        </button>
      )}

      {uploading && (
        <div style={{ marginTop: '20px' }}>
          <div style={{
            width: '100%',
            height: '6px',
            backgroundColor: '#e0e0e0',
            borderRadius: '3px',
            overflow: 'hidden'
          }}>
            <div style={{
              width: `${uploadProgress}%`,
              height: '100%',
              backgroundColor: '#2196f3',
              transition: 'width 0.3s ease'
            }} />
          </div>
          <p style={{ color: '#666', marginTop: '10px' }}>
            Uploading: {uploadProgress}%
          </p>
        </div>
      )}
    </div>
  );
}

export default UploadVideo;
