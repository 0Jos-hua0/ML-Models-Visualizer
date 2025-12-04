import React, { useState, useRef } from "react";
import Plot from "react-plotly.js";
import axios from "axios";

function ModelUploader({ setExplanation }) {
  const fileInputRef = useRef(null);
  const [plotData, setPlotData] = useState(null);
  const [imageData, setImageData] = useState(null);
  const [message, setMessage] = useState("");
  const [modelType, setModelType] = useState("");
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    const file = fileInputRef.current.files[0];
    if (!file) {
      setMessage("Please select a file first!");
      return;
    }

    setLoading(true);
    setMessage("Uploading and analyzing model...");

    const formData = new FormData();
    formData.append("model", file);

    try {
      const response = await axios.post("http://localhost:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const { message, model_type, explanation, plotly, image } = response.data;
      console.log("✅ Got response:", response.data);

      setMessage(message || "Model loaded successfully!");
      setModelType(model_type || "Unknown Model");

      // Handle explanation
      if (explanation) {
        setExplanation(explanation);
      } else {
        setExplanation(null);
      }

      // Clear previous visuals
      setPlotData(null);
      setImageData(null);

      // Handle visualizations
      if (plotly && plotly.data && plotly.layout) {
        setPlotData(plotly);
      } else if (image) {
        setImageData(image);
      }

      setLoading(false);

    } catch (err) {
      console.error("❌ Upload failed:", err.response ? err.response.data : err.message);
      const errorMsg = err.response?.data?.error || "Upload failed. Please try again.";
      setMessage(errorMsg);
      setPlotData(null);
      setImageData(null);
      setExplanation(null);
      setModelType("");
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "1.5rem" }}>
      <h2 style={{ marginBottom: "1rem", color: "#333" }}>Upload Your Trained Model</h2>

      <div style={{ marginBottom: "1rem" }}>
        <input
          type="file"
          ref={fileInputRef}
          accept=".pkl,.joblib,.pickle"
          style={{ marginRight: "1rem" }}
        />
        <button
          onClick={handleUpload}
          disabled={loading}
          style={{
            padding: "0.6rem 1.5rem",
            backgroundColor: loading ? "#6c757d" : "#007bff",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: loading ? "not-allowed" : "pointer",
            fontSize: "1rem",
            fontWeight: "500",
            transition: "background-color 0.3s"
          }}
        >
          {loading ? "Processing..." : "Upload & Analyze"}
        </button>
      </div>

      <p style={{ fontSize: "0.85rem", color: "#6c757d", marginTop: "0.5rem" }}>
        📁 Supported formats: <strong>.pkl</strong>, <strong>.joblib</strong>, <strong>.pickle</strong> (scikit-learn models)
      </p>

      {message && (
        <p style={{
          color: message.includes("success") || message.includes("loaded") ? "#28a745" : "#dc3545",
          fontWeight: "500",
          marginTop: "0.5rem"
        }}>
          {message}
        </p>
      )}

      {modelType && (
        <div style={{
          backgroundColor: "#e7f3ff",
          padding: "0.75rem 1rem",
          borderRadius: "5px",
          marginTop: "1rem",
          borderLeft: "4px solid #007bff"
        }}>
          <strong>Detected Model:</strong> {modelType}
        </div>
      )}

      {/* Plotly Visualization */}
      {plotData && (
        <div style={{ marginTop: "2rem" }}>
          <h3 style={{ color: "#333", marginBottom: "1rem" }}>📊 Model Visualization</h3>
          <div style={{
            backgroundColor: "white",
            padding: "1rem",
            borderRadius: "8px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.1)"
          }}>
            <Plot
              data={plotData.data}
              layout={{
                ...plotData.layout,
                autosize: true,
                margin: { l: 50, r: 50, t: 50, b: 50 }
              }}
              config={{ responsive: true, displayModeBar: true }}
              style={{ width: "100%", height: "500px" }}
            />
          </div>
        </div>
      )}

      {/* Matplotlib Image */}
      {imageData && (
        <div style={{ marginTop: "2rem" }}>
          <h3 style={{ color: "#333", marginBottom: "1rem" }}>🌳 Model Visualization</h3>
          <div style={{
            backgroundColor: "white",
            padding: "1rem",
            borderRadius: "8px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
            textAlign: "center"
          }}>
            <img
              src={`data:image/png;base64,${imageData}`}
              alt="Model Visualization"
              style={{
                maxWidth: "100%",
                height: "auto",
                borderRadius: "4px"
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default ModelUploader;
