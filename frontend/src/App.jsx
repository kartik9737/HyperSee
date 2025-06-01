import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import ModelResults from './components/ModelResults';
import DatasetSelector from './components/DatasetSelector';
import Toast from './components/Toast';
import PipelineSteps from './components/PipelineSteps';
import aLogo from './a.png';

function App() {
  const [result, setResult] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [backendStatus, setBackendStatus] = useState('Checking backend status...');
  const [toast, setToast] = useState(null);
  const [darkMode, setDarkMode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [activeSection, setActiveSection] = useState('home');

  const showToast = useCallback((message, type = 'info') => {
    setToast({ message, type });
  }, []);

  const handleToastClose = useCallback(() => {
    setToast(null);
  }, []);

  const toggleHelp = useCallback(() => {
    setShowHelp(prev => !prev);
    setShowShortcuts(false);
    showToast(showHelp ? 'Help panel closed' : 'Help panel opened', 'info');
  }, [showHelp, showToast]);

  const toggleShortcuts = useCallback(() => {
    setShowShortcuts(prev => !prev);
    setShowHelp(false);
    showToast(showShortcuts ? 'Shortcuts panel closed' : 'Shortcuts panel opened', 'info');
  }, [showShortcuts, showToast]);

  const toggleDarkMode = useCallback(() => {
    setDarkMode(prev => !prev);
    showToast(`Switched to ${darkMode ? 'light' : 'dark'} mode`, 'info');
  }, [darkMode, showToast]);

  useEffect(() => {
    const savedTheme = localStorage.getItem('darkMode');
    if (savedTheme !== null) {
      setDarkMode(JSON.parse(savedTheme));
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    document.body.classList.toggle('dark-mode', darkMode);
  }, [darkMode]);

  useEffect(() => {
    const handleKeyPress = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'd') {
        event.preventDefault();
        toggleDarkMode();
      }
      if (event.key === 'Escape' && toast) {
        handleToastClose();
      }
      if ((event.ctrlKey || event.metaKey) && event.key === 'h') {
        event.preventDefault();
        toggleHelp();
      }
      if ((event.ctrlKey || event.metaKey) && event.key === 's') {
        event.preventDefault();
        toggleShortcuts();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [toast, toggleDarkMode, handleToastClose, toggleHelp, toggleShortcuts]);

  const announceStatus = useCallback((message) => {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    document.body.appendChild(announcement);
    setTimeout(() => announcement.remove(), 1000);
  }, []);

  const checkBackendStatus = useCallback(async () => {
    let retries = 3;
    while (retries > 0) {
      try {
        const response = await fetch('http://127.0.0.1:5000/ping');
        const data = await response.json();
        if (response.ok) {
          setBackendStatus(data.message);
          announceStatus('Backend connection successful');
          return;
        }
      } catch (error) {
        retries--;
        if (retries === 0) {
          setBackendStatus('Error connecting to backend');
          announceStatus('Backend connection failed');
          showToast('Backend connection failed. Please check your connection.', 'error');
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }, [showToast, announceStatus]);

  useEffect(() => {
    checkBackendStatus();
    const intervalId = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(intervalId);
  }, [checkBackendStatus]);

  const callClassifyEndpoint = useCallback(async (hsi_path, gt_path, dataset_name) => {
    try {
      const response = await fetch('http://127.0.0.1:5000/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ hsi_path, gt_path, dataset_name })
      });
      const data = await response.json();
      if (response.ok) {
        setClassificationResult(data);
        showToast('Classification completed successfully', 'success');
      } else {
        showToast(data.error || 'Classification failed', 'error');
      }
    } catch (error) {
      showToast(error.message || 'Classification request failed', 'error');
    }
  }, [showToast]);

  const handleUploadSuccess = useCallback((results) => {
    if (results && results.images && selectedDataset) {
      const updatedImages = results.images.map(img => {
        if (img.name) {
          if (img.name.toLowerCase().includes('confusion matrix') || img.name.toLowerCase().includes('anomaly map')) {
            return {
              ...img,
              name: `${img.name} - ${selectedDataset}`
            };
          }
        }
        return img;
      });
      results.images = updatedImages;
    }
    setResult(results);
    showToast('Upload successful! Results are ready.', 'success');

    if (results.hsi_path && results.gt_path && selectedDataset) {
      callClassifyEndpoint(results.hsi_path, results.gt_path, selectedDataset);
    } else {
      showToast('Missing file paths or dataset name for classification', 'error');
    }
  }, [selectedDataset, callClassifyEndpoint, showToast]);

  const handleUploadFailure = useCallback((error) => {
    showToast(error?.message || 'Upload failed. Please try again.', 'error');
  }, [showToast]);

  const handleDatasetChange = useCallback((dataset) => {
    setSelectedDataset(dataset);
    showToast(`Dataset changed to: ${dataset}`, 'info');
  }, [showToast]);

  const goToHome = () => setActiveSection('home');
  const goToHowItWorks = () => setActiveSection('howItWorks');
  const goToProcessing = () => setActiveSection('processing');

  return (
    <div className={`App ${darkMode ? 'dark' : 'light'}`}>
      <header className="app-header">
        <div className="left-section">
          <img src={aLogo} alt="App Logo" className="app-logo" />
          <h1 className="app-title">AnomVisor</h1>
        </div>
        <nav className="center-nav">
          <button onClick={goToHome} className={activeSection === 'home' ? 'active' : ''}>Home</button>
          <button onClick={goToHowItWorks} className={activeSection === 'howItWorks' ? 'active' : ''}>How it Works</button>
          <button 
            className="toggle-dark-mode-btn" 
            onClick={toggleDarkMode}
            aria-label={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
            title="Toggle Dark Mode (Ctrl/Cmd + D)"
          >
            {darkMode ? '🌞 Light Mode' : '🌙 Dark Mode'}
          </button>
        </nav>
        <div className="right-section">
          <button onClick={goToProcessing} className={activeSection === 'processing' ? 'active' : ''}>Get Started</button>
        </div>
      </header>

      {activeSection === 'processing' && (
        <header>
          <h1>Hyperspectral Image Anomaly Detection</h1>
          <div className="header-info">
            <p className="status">
              <span className={`status-indicator ${backendStatus.includes('Error') ? 'error' : 'success'}`}></span>
              Backend status: {backendStatus}
            </p>
            <div className="system-info">
              <span className="separator">•</span>
              <span className="last-check">Last checked: {new Date().toLocaleTimeString()}</span>
            </div>
          </div>
        </header>
      )}

      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={handleToastClose}
        />
      )}

      <div className="container">
        <main>
          {activeSection === 'home' && (
            <section className="home-section">
              <div className="home-content" style={{ display: 'flex', flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                <div className="home-text" style={{ order: 1, flex: '1 1 50%' }}>
                  <h2>Welcome to Hyperspectral Image Anomaly Detection</h2>
                  <p><i>Hyperspectral Imaging (HSI) is a cutting-edge remote sensing technology that captures images across hundreds of narrow and contiguous spectral bands. Unlike traditional RGB imaging, which captures just three color channels (red, green, and blue), hyperspectral sensors gather detailed spectral information for each pixel, often across 100–300+ bands in the visible and infrared range.</i></p>
                  <p>HSI is widely used in various fields, including agriculture, environmental monitoring, and medical diagnostics. It enables the detection of subtle changes in materials and can identify specific chemical compositions.</p>
                  <p>In this application, we leverage advanced machine learning techniques to analyze hyperspectral images and detect anomalies. The process involves using an Autoencoder Transformer model for feature extraction and a Support Vector Machine (SVM) for classification.</p>

                </div>
                <div className="home-image" style={{ order: 2, flex: '1 1 50%', textAlign: 'right' }}>
                  {/* Using imported local image */}
                  <img
                    src={require('./hyperspectral-1406x1536.webp').default}
                    alt="Home Background"
                    style={{ maxHeight: '400px', width: 'auto', objectFit: 'contain', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)' }}
                  />
                </div>
              </div>
              <p className="home-description">This application allows you to upload hyperspectral images and detect anomalies using advanced machine learning models.</p>
              <p className="home-description">To explore,click on "Get Started" and select a dataset from the dropdown menu and upload your hyperspectral image file. The backend will process the image and display the results, including anomaly maps and metrics.</p>
              <p className="home-description">Explore the "How It Works" section to learn more about the underlying technology and methodology.</p>

            </section>
          )}

          {activeSection === 'howItWorks' && (
            <section className="how-it-works-section">
              <div className="home-content" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                <div style={{ flex: '1 1 100%' }}>
                  <h2>How It Works</h2>
                  <p>This application processes hyperspectral images through a series of steps:</p>
              <PipelineSteps isDarkMode={darkMode} />
                  
                </div>
              </div>
            </section>
          )}

          {activeSection === 'processing' && (
            <>
              <DatasetSelector
                selectedDataset={selectedDataset}
                onDatasetChange={handleDatasetChange}
              />
              <FileUpload
                selectedDataset={selectedDataset}
                onUploadSuccess={handleUploadSuccess}
                onUploadFailure={handleUploadFailure}
                isLoading={isLoading}
                setIsLoading={setIsLoading}
              />
              {isLoading ? (
                <div className="loading-container">
                  <div className="loading-spinner"></div>
                  <p>Processing your request...</p>
                </div>
              ) : (
                <>
                  <ModelResults result={result} datasetName={selectedDataset} />
                  {classificationResult && classificationResult.classification_image_url && (
                    <div className="classification-result">
                      <h3>Classification Result</h3>
                      <img
                        src={classificationResult.classification_image_url}
                        alt="Classification Result"
                        style={{ maxWidth: '100%', height: 'auto', border: '1px solid #ccc', borderRadius: '4px' }}
                      />
                    </div>
                  )}
                </>
              )}
            </>
          )}
        </main>
      </div>

      <footer>
        <div className="footer-content">
          <div className="footer-actions">
            <div className="footer-buttons">
              <button onClick={goToHome} className={activeSection === 'home' ? 'active' : ''}>Home</button>
              <button onClick={goToHowItWorks} className={activeSection === 'howItWorks' ? 'active' : ''}>How it Works</button>
              <button onClick={goToProcessing} className={activeSection === 'processing' ? 'active' : ''}>Getting Started</button>
              <button onClick={toggleHelp} className={showHelp ? 'active' : ''}>Help</button>
              <button onClick={toggleShortcuts} className={showShortcuts ? 'active' : ''}>Shortcuts</button>
            </div>
            <button 
              className="toggle-dark-mode-btn" 
              onClick={toggleDarkMode}
              aria-label={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
              title="Toggle Dark Mode (Ctrl/Cmd + D)"
            >
              {darkMode ? '🌞 Light Mode' : '🌙 Dark Mode'}
            </button>
          </div>
          <div className="team-info">
            <p className="team-name">AnomVisor</p>
            <p className="copyright">© {new Date().getFullYear()} All rights reserved</p>
          </div>
        </div>
      </footer>

      {showHelp && (
        <div className="help-panel">
          <h2>Help</h2>
          <div className="help-content">
            <section>
              <h3>Getting Started</h3>
              <p>Welcome to the Hyperspectral Image Anomaly Detection application. Here's how to use it:</p>
              <ul>
                <li>Select a dataset from the dropdown menu</li>
                <li>Upload your hyperspectral image file</li>
                <li>Wait for the processing to complete</li>
                <li>View the results in the table below</li>
              </ul>
            </section>
            <section>
              <h3>Features</h3>
              <ul>
                <li>Dark/Light mode toggle</li>
                <li>Keyboard shortcuts for quick access</li>
                <li>Real-time backend status monitoring</li>
                <li>Toast notifications for important updates</li>
              </ul>
            </section>
          </div>
          <button className="close-btn" onClick={toggleHelp}>Close</button>
        </div>
      )}

      {showShortcuts && (
        <div className="shortcuts-panel">
          <h2>Keyboard Shortcuts</h2>
          <div className="shortcuts-content">
            <div className="shortcut-item">
              <kbd>Ctrl/Cmd</kbd> + <kbd>D</kbd>
              <span>Toggle Dark Mode</span>
            </div>
            <div className="shortcut-item">
              <kbd>Esc</kbd>
              <span>Close Toast Notifications</span>
            </div>
            <div className="shortcut-item">
              <kbd>Ctrl/Cmd</kbd> + <kbd>H</kbd>
              <span>Show/Hide Help</span>
            </div>
            <div className="shortcut-item">
              <kbd>Ctrl/Cmd</kbd> + <kbd>S</kbd>
              <span>Show/Hide Shortcuts</span>
            </div>
          </div>
          <button className="close-btn" onClick={toggleShortcuts}>Close</button>
        </div>
      )}
    </div>
  );
}

export default App;

