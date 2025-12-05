import React, { useState, useRef } from "react";
import Plot from "react-plotly.js";
import axios from "axios";

// Import icons
import uploadIcon from "../assets/icon/upload.png";
import analyzeIcon from "../assets/icon/analyze.png";

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

      if (explanation) {
        setExplanation(explanation);
      } else {
        setExplanation(null);
      }

      setPlotData(null);
      setImageData(null);

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
    <div style={{ padding: "2.5rem" }}>
      {/* Upload Controls */}
      <div style={{
        backgroundColor: "#c1bdb3", // Silver
        padding: "2.5rem",
        borderRadius: "8px",
        border: "2px dashed #2e6171", // Blue Slate
        textAlign: "center",
        marginBottom: "2rem"
      }}>
        <div style={{
          marginBottom: "1rem",
          display: "flex",
          justifyContent: "center"
        }}>
          <img
            src={uploadIcon}
            alt="Upload"
            style={{ width: "120px", height: "120px", objectFit: "contain" }}
          />
        </div>

        <input
          type="file"
          ref={fileInputRef}
          accept=".pkl,.joblib,.pickle"
          style={{
            display: "none"
          }}
          id="file-upload"
        />

        <label
          htmlFor="file-upload"
          style={{
            display: "inline-block",
            padding: "0.8rem 2.5rem",
            backgroundColor: "#2e6171", // Blue Slate
            color: "#ffffff",
            borderRadius: "4px",
            cursor: "pointer",
            fontSize: "1rem",
            fontWeight: "600",
            marginBottom: "1.5rem",
            transition: "all 0.3s ease",
            border: "2px solid #2e6171"
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = "#1a3c48";
            e.currentTarget.style.borderColor = "#1a3c48";
            e.currentTarget.style.transform = "translateY(-2px)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = "#2e6171";
            e.currentTarget.style.borderColor = "#2e6171";
            e.currentTarget.style.transform = "translateY(0)";
          }}
        >
          Choose Model File
        </label>

        <div style={{
          fontSize: "1rem",
          color: "#000000",
          marginBottom: "1.5rem",
          fontWeight: "500"
        }}>
          {fileInputRef.current?.files[0]?.name || "No file selected"}
        </div>

        <button
          onClick={handleUpload}
          disabled={loading}
          style={{
            padding: "0.8rem 3rem",
            backgroundColor: loading ? "#78909c" : "#a5402d", // Reddish Brown
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: loading ? "not-allowed" : "pointer",
            fontSize: "1.05rem",
            fontWeight: "700",
            transition: "all 0.3s ease",
            boxShadow: loading ? "none" : "0 4px 6px rgba(165, 64, 45, 0.3)"
          }}
          onMouseEnter={(e) => {
            if (!loading) {
              e.currentTarget.style.backgroundColor = "#8a3525";
              e.currentTarget.style.transform = "translateY(-2px)";
              e.currentTarget.style.boxShadow = "0 6px 12px rgba(165, 64, 45, 0.4)";
            }
          }}
          onMouseLeave={(e) => {
            if (!loading) {
              e.currentTarget.style.backgroundColor = "#a5402d";
              e.currentTarget.style.transform = "translateY(0)";
              e.currentTarget.style.boxShadow = "0 4px 6px rgba(165, 64, 45, 0.3)";
            }
          }}
        >
          {loading ? "⏳ Processing..." : "🚀 Analyze Model"}
        </button>

        <p style={{
          fontSize: "0.9rem",
          color: "#2e6171",
          marginTop: "1.5rem",
          marginBottom: 0,
          fontWeight: "500"
        }}>
          Supported: <strong>.pkl</strong>, <strong>.joblib</strong>, <strong>.pickle</strong>
        </p>
      </div>

      {/* Status Message */}
      {message && (
        <div style={{
          padding: "1rem 1.5rem",
          borderRadius: "4px",
          marginBottom: "1.5rem",
          backgroundColor: message.includes("success") || message.includes("loaded")
            ? "#e8f5e9"
            : message.includes("Uploading")
              ? "#e3f2fd"
              : "#ffebee",
          borderLeft: `5px solid ${message.includes("success") || message.includes("loaded")
              ? "#2e7d32"
              : message.includes("Uploading")
                ? "#2e6171"
                : "#c62828"
            }`,
          color: "#000000",
          fontWeight: "600",
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          boxShadow: "0 2px 4px rgba(0,0,0,0.05)"
        }}>
          <span style={{ fontSize: "1.2rem" }}>
            {message.includes("success") || message.includes("loaded")
              ? "✅"
              : message.includes("Uploading")
                ? "⏳"
                : "❌"}
          </span>
          <span>{message}</span>
        </div>
      )}

      {/* Model Type Badge */}
      {modelType && (
        <div style={{
          backgroundColor: "#c1bdb3", // Silver
          padding: "1.5rem",
          borderRadius: "4px",
          marginBottom: "1.5rem",
          border: "2px solid #2e6171",
          display: "flex",
          alignItems: "center",
          gap: "1rem"
        }}>
          <div style={{ width: "64px", height: "64px" }}>
            <img
              src={analyzeIcon}
              alt="Model Type"
              style={{ width: "100%", height: "100%", objectFit: "contain" }}
            />
          </div>
          <div>
            <div style={{ fontSize: "0.9rem", color: "#2e6171", fontWeight: "600", textTransform: "uppercase" }}>
              Detected Model Type
            </div>
            <div style={{ fontSize: "1.4rem", color: "#000000", fontWeight: "800" }}>
              {modelType}
            </div>
          </div>
        </div>
      )}

      {/* Plotly Visualization */}
      {plotData && (
        <div style={{ marginTop: "2.5rem" }}>
          <div style={{
            backgroundColor: "#2e6171", // Blue Slate
            padding: "1rem 1.5rem",
            borderRadius: "4px 4px 0 0",
            color: "white",
            fontWeight: "700",
            fontSize: "1.2rem",
            display: "flex",
            alignItems: "center",
            gap: "0.8rem",
            borderBottom: "3px solid #a5402d"
          }}>
            <span>📊</span>
            <span>Interactive Visualization</span>
          </div>
          <div style={{
            backgroundColor: "white",
            padding: "2rem",
            borderRadius: "0 0 4px 4px",
            border: "1px solid #2e6171",
            borderTop: "none"
          }}>
            <Plot
              data={plotData.data}
              layout={{
                ...plotData.layout,
                autosize: true,
                margin: { l: 60, r: 60, t: 60, b: 60 },
                paper_bgcolor: "#ffffff",
                plot_bgcolor: "#ffffff"
              }}
              config={{ responsive: true, displayModeBar: true }}
              style={{ width: "100%", height: "600px" }}
            />
          </div>
        </div>
      )}

      {/* Matplotlib Image */}
      {imageData && (
        <div style={{ marginTop: "2.5rem" }}>
          <div style={{
            backgroundColor: "#2e6171",
            padding: "1rem 1.5rem",
            borderRadius: "4px 4px 0 0",
            color: "white",
            fontWeight: "700",
            fontSize: "1.2rem",
            display: "flex",
            alignItems: "center",
            gap: "0.8rem",
            borderBottom: "3px solid #a5402d"
          }}>
            <span>🌳</span>
            <span>Model Structure</span>
          </div>
          <div style={{
            backgroundColor: "white",
            padding: "2rem",
            borderRadius: "0 0 4px 4px",
            border: "1px solid #2e6171",
            borderTop: "none",
            textAlign: "center"
          }}>
            <img
              src={`data:image/png;base64,${imageData}`}
              alt="Model Visualization"
              style={{
                maxWidth: "100%",
                height: "auto",
                borderRadius: "4px",
                border: "2px solid #c1bdb3"
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default ModelUploader;
