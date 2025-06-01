import React from 'react';

function DatasetSelector({ selectedDataset, onDatasetChange }) {
  return (
    <div className="dataset-selector">
      <label htmlFor="dataset-select" className="dataset-label">
        Select Dataset:
      </label>
      <select
        id="dataset-select"
        className="dataset-dropdown"
        value={selectedDataset}
        onChange={(e) => onDatasetChange(e.target.value)}
      >
        <option value="">-- Choose a Dataset --</option>
        <option value="salinas">Salinas</option>
        <option value="indian">Indian Pines</option>
        <option value="pavia">Pavia University</option>
      </select>
    </div>
  );
}

export default DatasetSelector;
