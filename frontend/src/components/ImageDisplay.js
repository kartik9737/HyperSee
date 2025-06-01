import React, { useState } from 'react';
import './ImageDisplay.css';

function ImageDisplay({ images, title }) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);

  const handleImageClick = (image) => {
    setSelectedImage(image);
    setZoomLevel(1);
  };

  const handleClose = () => {
    setSelectedImage(null);
  };

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev + 0.1, 3));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev - 0.1, 0.5));
  };

  const handleDownload = (imageUrl, imageName) => {
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = imageName || 'hyperspectral-image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!images || images.length === 0) {
    return null;
  }

  return (
    <div className="image-display-container">
      <h3>{title || 'Output Images'}</h3>
      <div className="image-grid">
        {images.map((image, index) => (
          <div key={index} className="image-item">
            <img
              src={image.url}
              alt={image.name || `Output ${index + 1}`}
              onClick={() => handleImageClick(image)}
              className="thumbnail"
            />
            <div className="image-info">
              <span className="image-name">{image.name || `Image ${index + 1}`}</span>
              <button
                className="download-btn"
                onClick={() => handleDownload(image.url, image.name)}
                title="Download Image"
              >
                ⬇️ Download
              </button>
            </div>
          </div>
        ))}
      </div>

      {selectedImage && (
        <div className="image-modal">
          <div className="modal-content">
            <div className="modal-header">
              <h3>{selectedImage.name || 'Image Preview'}</h3>
              <div className="modal-controls">
                <button onClick={handleZoomOut} title="Zoom Out">-</button>
                <span>{Math.round(zoomLevel * 100)}%</span>
                <button onClick={handleZoomIn} title="Zoom In">+</button>
                <button onClick={handleClose} className="close-btn" title="Close">×</button>
              </div>
            </div>
            <div className="modal-body">
              <img
                src={selectedImage.url}
                alt={selectedImage.name || 'Preview'}
                style={{ transform: `scale(${zoomLevel})` }}
              />
            </div>
            <div className="modal-footer">
              <button
                className="download-btn"
                onClick={() => handleDownload(selectedImage.url, selectedImage.name)}
              >
                ⬇️ Download Image
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageDisplay; 