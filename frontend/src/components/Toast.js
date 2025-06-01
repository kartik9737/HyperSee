// Toast.js
import React, { useEffect, useState } from 'react';
import ReactDOM from 'react-dom';
import './Toast.css';

const Toast = ({ message, type, onClose }) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    // Show toast immediately
    setIsVisible(true);

    // Start hiding after 3.5 seconds
    const hideTimer = setTimeout(() => {
      setIsVisible(false);
    }, 3500);

    // Remove from DOM after fade out
    const removeTimer = setTimeout(() => {
      onClose?.();
    }, 4000);

    return () => {
      clearTimeout(hideTimer);
      clearTimeout(removeTimer);
    };
  }, [onClose]);

  const toastContent = (
    <div className={`toast ${type} ${isVisible ? 'toast-show' : 'toast-hide'}`}>
      <div className="toast-content">
        <span className="toast-icon">
          {type === 'success' && '✓'}
          {type === 'error' && '✕'}
          {type === 'info' && 'ℹ'}
        </span>
        <span className="toast-message">{message}</span>
      </div>
      <div className="toast-progress"></div>
    </div>
  );

  return ReactDOM.createPortal(
    toastContent,
    document.body
  );
};

export default Toast;
