import React, { useState, useRef } from "react";
import Plot from "react-plotly.js";
import axios from "axios";

function ModelUploader() {
  const fileInputRef = useRef(null);
  const [plotData, setPlotData] = useState(null);
  const [imageData, setImageData] = useState(null);
  const [reportText, setReportText] = useState("");
  const [reportSummary, setReportSummary] = useState(null); // 🆕 Summary State
  const [message, setMessage] = useState("");

  const handleUpload = async () => {
    const file = fileInputRef.current.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("model", file);

    try {
      const response = await axios.post("http://localhost:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const { message, report, report_summary, plotly, image } = response.data;
      console.log("✅ Got response:", response.data);
      setMessage(message || "");

      // Handle full report
      if (report) {
        const formattedReport = JSON.stringify(report, null, 2);
        setReportText(formattedReport);
        console.log("🧾 Set report:", formattedReport);
      } else {
        setReportText("No report available.");
      }

      // Handle report summary
      if (report_summary) {
        setReportSummary(report_summary);
      } else {
        setReportSummary(null);
      }

      // Clear visuals first
      setPlotData(null);
      setImageData(null);

      // Handle visuals
      if (plotly && plotly.data && plotly.layout) {
        setPlotData(plotly);
      } else if (image) {
        if (image.data && image.layout) {
          setPlotData(image);
        } else if (typeof image === "string") {
          setImageData(image);
        }
      }

    } catch (err) {
      console.error("❌ Upload failed:", err.response ? err.response.data : err.message);
      setMessage("Upload failed.");
      setPlotData(null);
      setImageData(null);
      setReportText("");
      setReportSummary(null);
    }
  };

  return (
    <div style={{ padding: "1rem" }}>
      <h2>Upload Trained Model</h2>
      <input type="file" ref={fileInputRef} />
      <button
        onClick={handleUpload}
        style={{
          marginLeft: "1rem",
          padding: "0.5rem 1rem",
          backgroundColor: "#007bff",
          color: "white",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
        }}
      >
        Upload Model
      </button>

      <p>{message}</p>

      {/* Report Summary */}
      {reportSummary && (
        <div style={{ marginTop: "2rem" }}>
          <h3>📊 Model Report Summary</h3>
          <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <tbody>
              {Object.entries(reportSummary).map(([key, value]) => (
                <tr key={key}>
                  <td style={{ padding: "8px", border: "1px solid #ccc", fontWeight: "bold" }}>{key}</td>
                  <td style={{ padding: "8px", border: "1px solid #ccc" }}>
  <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(value)}</pre>
</td>

                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Plotly Visualization */}
      {plotData && (
        <div style={{ marginTop: "2rem" }}>
          <h3>📈 Model Visualization (Plotly)</h3>
          <Plot
            data={plotData.data}
            layout={plotData.layout}
            config={{ responsive: true }}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
      )}

      {/* Matplotlib Image */}
      {imageData && (
        <div style={{ marginTop: "2rem" }}>
          <h3>🖼️ Model Visualization (Matplotlib)</h3>
          <img
            src={`data:image/png;base64,${imageData}`}
            alt="Model Visualization"
            style={{ maxWidth: "100%", border: "1px solid #ccc" }}
          />
        </div>
      )}

      {/* Full Report */}
{reportText && (
  <div style={{ marginTop: "2rem" }}>
    <h3>📄 Full Model Report</h3>
    <table style={{ borderCollapse: "collapse", width: "100%" }}>
      <tbody>
        {Object.entries(JSON.parse(reportText)).map(([key, value]) => (
          <tr key={key}>
            <td style={{ padding: "8px", border: "1px solid #ccc", fontWeight: "bold", verticalAlign: "top" }}>{key}</td>
            <td style={{ padding: "8px", border: "1px solid #ccc" }}>
              {typeof value === "object"
                ? <pre style={{ margin: 0 }}>{JSON.stringify(value, null, 2)}</pre>
                : String(value)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  </div>
)}

    </div>
  );
}

export default ModelUploader;
