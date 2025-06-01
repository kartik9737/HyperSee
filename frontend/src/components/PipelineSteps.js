import React from 'react';
import './PipelineSteps.css';

const Arrow = () => (
  <div className="arrow">
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
      <path d="M4 12h16M16 6l6 6-6 6" stroke="#333" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  </div>
);

const PipelineSteps = ({ isDarkMode }) => {
  return (
    <div className={`pipeline-steps-container${isDarkMode ? ' dark' : ''}`}>
      <div className="step-box">
        <h3>Step 1: Select a Dataset</h3>
        <p>
          Begin by choosing a predefined hyperspectral dataset (e.g., Indian Pines, Salinas, Pavia University).
          The selected dataset determines the spatial and spectral characteristics of the analysis and sets up 
          appropriate parameters for further processing.
        </p>
      </div>
      <Arrow />
      <div className="step-box">
        <h3>Step 2: Upload Image Files</h3>
        <p>
          Upload your own hyperspectral data file (typically in `.mat` format) along with its corresponding ground 
          truth label file. These files are essential for training, validating, and benchmarking the anomaly detection pipeline.
        </p>
      </div>
      <Arrow />
      <div className="step-box">
        <h3>Step 3: Backend Processing</h3>
        <p>
          The uploaded data is processed by a deep learning pipeline that includes:
          <br />• <strong>Autoencoder-Transformer:</strong> For efficient spectral-spatial feature extraction.
          <br />• <strong>Support Vector Machine (SVM):</strong> For robust classification based on the extracted features.
          <br />• Preprocessing includes normalization, optional PCA, and patch extraction.
        </p>
      </div>
      <Arrow />
      <div className="step-box">
        <h3>Step 4: View Results</h3>
        <p>
          Once processing is complete, results are visualized:
          <br />• Anomaly maps overlaid on RGB or PCA-reduced imagery.
          <br />• Evaluation metrics such as AUC, accuracy, precision, and recall.
          <br />• Class-specific insights for targeted anomaly detection.
        </p>
      </div>
    </div>
  );
};

export default PipelineSteps;
