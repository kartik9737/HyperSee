import React, { useState } from 'react';
import ImageDisplay from './ImageDisplay';
import './ModelResults.css';

function ModelResults({ result, datasetName }) {
  const [classificationImageUrl, setClassificationImageUrl] = useState(null);
  const [loadingClassification, setLoadingClassification] = useState(false);
  const [classificationError, setClassificationError] = useState(null);

  if (!result) return null;

  // Filter out anomaly histogram and anomaly score distribution images
  const images = result.images ? result.images
    .filter(img => img.name !== 'Anomaly Score Distribution' && img.name !== 'Anomaly Intensity Histogram')
    .map((img, index) => {
      let url = img.url || img.path || '';
      // Prepend backend base URL if url is relative and starts with /uploads/
      if (url.startsWith('/uploads/')) {
        url = `http://localhost:5000${url}`;
      }
      return {
        url,
        name: img.name || `Result ${index + 1}`,
        description: img.description
      };
    }) : [];

  const getFilename = (path) => {
    if (!path) return '';
    return path.split('/').pop();
  };

  const handleClassifyClick = async () => {
    setLoadingClassification(true);
    setClassificationError(null);
    setClassificationImageUrl(null);
    try {
      const response = await fetch('http://localhost:5000/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hsi_path: getFilename(result.hsi_path),
          gt_path: getFilename(result.gt_path),
          dataset_name: datasetName
        })
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Server error: ${text}`);
      }
      const data = await response.json();
      setClassificationImageUrl(data.classification_image_url);
    } catch (error) {
      setClassificationError(error.message || 'Classification failed');
    } finally {
      setLoadingClassification(false);
    }
  };

  const accuracy = result.stats && typeof result.stats.accuracy === 'number' ? result.stats.accuracy : null;

  return (
    <div className="model-results">
      <h2>Analysis Results</h2>
      
      {/* Display any numerical results or statistics */}
      {accuracy !== null && (
        <div className="results-stats">
          <h3>Statistics</h3>
          <div className="stats-section">
            <h4>Overall Accuracy</h4>
            <p>{(accuracy * 100).toFixed(2) + '%'}</p>
          </div>
        </div>
      )}

      {/* Display output images */}
      {images.length > 0 && (
        <ImageDisplay
          images={images}
          title="Analysis Output Images"
        />
      )}

      {/* Classify the anomalies button, shown only if detection results exist */}
      {images.length > 0 && (
        <div className="classification-section">
          <button onClick={handleClassifyClick} disabled={loadingClassification}>
            {loadingClassification ? 'Classifying...' : 'Classify the Anomalies'}
          </button>
          {classificationError && <p className="error-message">{classificationError}</p>}
          
        </div>
      )}

      {/* Display any additional information */}
      {result.info && (
        <div className="results-info">
          <h3>Additional Information</h3>
          <p>{result.info}</p>
        </div>
      )}
    </div>
  );
}

export default ModelResults;
