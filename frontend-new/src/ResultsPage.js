import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function ResultsPage({ videoId }) {
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        const res = await axios.get(`http://localhost:8000/results/${videoId}`);
        setResults(res.data);
      } catch (error) {
        setError(error.message);
        console.error("Failed to load results:", error);
      }
    };
    fetchResults();
  }, [videoId]);

  const MetricCard = ({ title, value, icon, color }) => (
    <div style={{
      backgroundColor: 'white',
      borderRadius: '10px',
      padding: '20px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      flex: '1',
      minWidth: '200px',
      margin: '10px'
    }}>
      <div style={{ fontSize: '24px', marginBottom: '10px' }}>{icon}</div>
      <h3 style={{ margin: '0 0 10px 0', color: '#333' }}>{title}</h3>
      <div style={{ 
        fontSize: '24px', 
        fontWeight: 'bold',
        color: color
      }}>
        {typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : value}
      </div>
    </div>
  );

  const TimeSeriesChart = ({ data, labels, title, color }) => {
    const chartData = {
      labels: labels,
      datasets: [
        {
          label: title,
          data: data,
          borderColor: color,
          backgroundColor: color + '20',
          fill: true,
          tension: 0.4,
        },
      ],
    };

    const options = {
      responsive: true,
      plugins: {
        legend: {
          display: false,
        },
        title: {
          display: true,
          text: title,
          color: '#333',
          font: {
            size: 16,
            weight: 'bold',
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          ticks: {
            callback: (value) => `${(value * 100)}%`,
          },
        },
        x: {
          title: {
            display: true,
            text: 'Time (seconds)',
          },
        },
      },
    };

    return <Line data={chartData} options={options} />;
  };

  if (error) {
    return (
      <div style={{ 
        padding: '20px',
        backgroundColor: '#ffebee',
        borderRadius: '10px',
        color: '#c62828',
        textAlign: 'center'
      }}>
        <h3>Error Loading Results</h3>
        <p>{error}</p>
      </div>
    );
  }

  if (!results) {
    return (
      <div style={{
        textAlign: 'center',
        padding: '40px'
      }}>
        <div style={{ fontSize: '48px', marginBottom: '20px' }}>⏳</div>
        <p>Loading your results...</p>
      </div>
    );
  }

  const average = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;

  // Calculate time segments where metrics were consistently good or needed improvement
  const analyzeTimeSegments = (scores, threshold = 0.7) => {
    let segments = [];
    let currentSegment = { start: 0, type: scores[0] >= threshold ? 'good' : 'improvement' };
    
    scores.forEach((score, i) => {
      const isGood = score >= threshold;
      const segmentType = isGood ? 'good' : 'improvement';
      
      if (segmentType !== currentSegment.type) {
        currentSegment.end = results.timestamps[i];
        segments.push(currentSegment);
        currentSegment = { start: results.timestamps[i], type: segmentType };
      }
    });
    
    // Add the last segment
    currentSegment.end = results.timestamps[results.timestamps.length - 1];
    segments.push(currentSegment);
    
    return segments;
  };

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '40px 20px'
    }}>
      {/* Header Section */}
      <div style={{
        textAlign: 'center',
        marginBottom: '40px'
      }}>
       {/* <h1 style={{ color: '#333', marginBottom: '10px' }}>Analysis Results</h1>*/}
        {/*<p style={{ color: '#666' }}>
          Video Duration: {results.summary.duration_sec.toFixed(1)} seconds
        </p>*/}
      </div>

      {/* Metrics Cards */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '20px',
        marginBottom: '40px',
        justifyContent: 'center'
      }}>
        <MetricCard
          title="Overall Performance"
          value={average(results.final_scores)}
          icon="🎯"
          color="#2196f3"
        />
        <MetricCard
          title="Smile Score"
          value={average(results.smile_scores)}
          icon="😊"
          color="#4caf50"
        />
        <MetricCard
          title="Eye Contact"
          value={average(results.eye_contact_scores)}
          icon="👁️"
          color="#9c27b0"
        />
        <MetricCard
          title="Posture Score"
          value={results.summary.avg_confidence}
          icon="🧍"
          color="#ff9800"
        />
      </div>

      {/* Time Series Charts */}
      <div style={{
        backgroundColor: 'white',
        borderRadius: '10px',
        padding: '20px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        marginBottom: '40px'
      }}>
        <h2 style={{ textAlign: 'center', marginBottom: '30px', color: '#333' }}>
          Performance Over Time
        </h2>
        <div style={{ marginBottom: '30px' }}>
          <TimeSeriesChart
            data={results.final_scores}
            labels={results.timestamps}
            title="Overall Performance"
            color="#2196f3"
          />
        </div>
        {/* <div style={{ marginBottom: '30px' }}>
          <TimeSeriesChart
            data={results.smile_scores}
            labels={results.timestamps}
            title="Smile Detection"
            color="#4caf50"
          />
        </div>
        <div style={{ marginBottom: '30px' }}>
          <TimeSeriesChart
            data={results.eye_contact_scores}
            labels={results.timestamps}
            title="Eye Contact"
            color="#9c27b0"
          />
        </div>
        <div>
          <TimeSeriesChart
            data={results.head_pose_scores}
            labels={results.timestamps}
            title="Head Pose Stability"
            color="#ff9800"
          />
        </div>
      </div>

      {/* Detailed Analysis */}
     </div> <div style={{
        backgroundColor: 'white',
        borderRadius: '10px',
        padding: '20px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
        marginBottom: '40px'
      }}>
       {/* <h2 style={{ textAlign: 'center', marginBottom: '30px', color: '#333' }}>
          Detailed Analysis Report
        </h2>
        {/*
        <div style={{ marginBottom: '30px' }}>
          <h3 style={{ color: '#333', marginBottom: '15px' }}>Key Observations</h3>
          <ul style={{ color: '#666', lineHeight: '1.6' }}>
            <li>Overall Performance: {average(results.final_scores) >= 0.7 ? 'Strong' : 'Needs Improvement'}</li>
            <li>Smile Engagement: {average(results.smile_scores) >= 0.7 ? 'Natural and Consistent' : 'Could be more frequent'}</li>
            <li>Eye Contact: {average(results.eye_contact_scores) >= 0.7 ? 'Well Maintained' : 'Needs more consistency'}</li>
            <li>Posture: {results.summary.avg_confidence >= 0.7 ? 'Confident and Stable' : 'Could be more confident'}</li>
          </ul>
        </div>

        <div style={{ marginBottom: '30px' }}>
          <h3 style={{ color: '#333', marginBottom: '15px' }}>Time-Based Analysis</h3>
          <div style={{ color: '#666', lineHeight: '1.6' }}>
            {analyzeTimeSegments(results.final_scores).map((segment, index) => (
              <p key={index}>
                {segment.type === 'good' ? '✅' : '💡'} {segment.type === 'good' ? 'Strong performance' : 'Room for improvement'} 
                from {segment.start.toFixed(1)}s to {segment.end.toFixed(1)}s
                ({((segment.end - segment.start) / results.summary.duration_sec * 100).toFixed(1)}% of video)
              </p>
            ))}
          </div>
        </div>*/}

        <div>
          <h3 style={{ color: '#333', marginBottom: '15px' }}>Recommendations</h3>
          <ul style={{ color: '#666', lineHeight: '1.6' }}>
            {average(results.smile_scores) < 0.7 && (
              <li>Work on maintaining a more natural smile throughout your presentation. Try practicing with more relaxed facial expressions.</li>
            )}
            {average(results.eye_contact_scores) < 0.7 && (
              <li>Improve eye contact by focusing on the camera more consistently. Practice speaking while maintaining eye contact with a fixed point.</li>
            )}
            {results.summary.avg_confidence < 0.7 && (
              <li>Enhance your posture by standing straight and keeping your shoulders back. Consider practicing in front of a mirror.</li>
            )}
            {average(results.head_pose_scores) < 0.7 && (
              <li>Keep your head more stable and facing forward. Try to minimize excessive head movements while speaking.</li>
            )}
          </ul>
        </div>
      </div>

      {/* Video Playback */}
      <div style={{
        backgroundColor: 'white',
        borderRadius: '10px',
        padding: '20px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <h2 style={{ 
          color: '#333',
          marginBottom: '20px',
          textAlign: 'center'
        }}>
          Analyzed Video
        </h2>
        <div style={{
          position: 'relative',
          paddingBottom: '56.25%',
          height: 0,
          overflow: 'hidden',
          borderRadius: '10px'
        }}>
          <video 
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              backgroundColor: '#f5f5f5'
            }}
            controls 
            autoPlay 
            muted 
            src={`http://localhost:8000/video/${videoId.replace(".mp4", "_analyzed.mp4")}`}
          />
        </div>
      </div>
    </div>
  );
}

export default ResultsPage;


