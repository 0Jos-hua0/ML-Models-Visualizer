import React, { useState } from "react";
import ModelUploader from "./components/ModelUploader";
import starIcon from "./assets/star.webp";
import backgroundImage from "./assets/NIGHT.webp";

function App() {
  const [reportText, setReportText] = useState("");

  return (
    <div
      style={{
        textAlign: "center",
        padding: "2rem",
        minHeight: "100vh",
        fontFamily: "'Segoe UI', sans-serif",
        backgroundImage: `url(${backgroundImage})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        color: "#fff",
      }}
    >
      {/* Header with stars */}
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
        <img
          src={starIcon}
          alt="star"
          style={{ width: "40px", height: "40px", marginRight: "10px" }}
        />
        <h1
          style={{
            fontSize: "3rem",
            marginBottom: "1rem",
            fontFamily: "'Dancing Script', cursive",
            background: "linear-gradient(to right, #add8e6, #ffb6c1)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            textShadow: "1px 1px 2px rgba(0,0,0,0.2)",
          }}
        >
          STELLARBASIN
        </h1>
        <img
          src={starIcon}
          alt="star"
          style={{ width: "40px", height: "40px", marginLeft: "10px" }}
        />
      </div>

      <p style={{ color: "#eee", fontSize: "1.1rem", marginBottom: "2rem" }}>
        Upload your trained ML model to see a smart visualization & report
      </p>

      {/* Upload section boxed */}
      <div
        style={{
          background: "white",
          color: "#222",
          borderRadius: "12px",
          padding: "2rem",
          maxWidth: "800px",
          margin: "0 auto 2rem auto",
          boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
        }}
      >
        <ModelUploader setReportText={setReportText} />
      </div>

      {/* Report section boxed */}
      {reportText && (
        <div
          style={{
            background: "white",
            color: "#222",
            borderRadius: "12px",
            padding: "2rem",
            maxWidth: "900px",
            margin: "2rem auto",
            boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
            textAlign: "left",
          }}
        >
          <h2 style={{ fontSize: "1.8rem", marginBottom: "1rem" }}>
            📄 Model Report Summary
          </h2>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              wordWrap: "break-word",
              background: "#f9f9f9",
              padding: "1rem",
              borderRadius: "8px",
              overflowX: "auto",
              fontSize: "0.95rem",
              color: "#444",
              fontFamily: "Consolas, monospace",
            }}
          >
            {reportText}
          </pre>
        </div>
      )}
    </div>
  );
}

export default App;
