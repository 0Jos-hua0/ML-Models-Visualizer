import React, { useState } from "react";
import ModelUploader from "./components/ModelUploader";
import ExplanationPanel from "./components/ExplanationPanel";
import starIcon from "./assets/star.webp";
import backgroundImage from "./assets/NIGHT.webp";

function App() {
  const [explanation, setExplanation] = useState(null);

  return (
    <div
      style={{
        minHeight: "100vh",
        fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        backgroundImage: `url(${backgroundImage})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        backgroundAttachment: "fixed",
        padding: "2rem 1rem",
      }}
    >
      {/* Header */}
      <div style={{
        textAlign: "center",
        marginBottom: "2rem",
        animation: "fadeIn 1s ease-in"
      }}>
        <div style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          marginBottom: "1rem"
        }}>
          <img
            src={starIcon}
            alt="star"
            style={{
              width: "50px",
              height: "50px",
              marginRight: "15px",
              animation: "rotate 20s linear infinite"
            }}
          />
          <h1
            style={{
              fontSize: "3.5rem",
              margin: 0,
              fontFamily: "'Dancing Script', cursive",
              background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              textShadow: "2px 2px 4px rgba(0,0,0,0.3)",
              fontWeight: "bold"
            }}
          >
            STELLARBASIN
          </h1>
          <img
            src={starIcon}
            alt="star"
            style={{
              width: "50px",
              height: "50px",
              marginLeft: "15px",
              animation: "rotate 20s linear infinite reverse"
            }}
          />
        </div>

        <p style={{
          color: "#fff",
          fontSize: "1.3rem",
          textShadow: "1px 1px 3px rgba(0,0,0,0.5)",
          maxWidth: "800px",
          margin: "0 auto",
          lineHeight: "1.6"
        }}>
          🚀 Advanced ML Model Visualizer & Explainer
        </p>

        <p style={{
          color: "#e0e0e0",
          fontSize: "1rem",
          textShadow: "1px 1px 2px rgba(0,0,0,0.5)",
          marginTop: "0.5rem"
        }}>
          Upload your trained models and get instant visualizations with detailed explanations
        </p>
      </div>

      {/* Supported Models Info */}
      <div style={{
        backgroundColor: "rgba(255, 255, 255, 0.95)",
        borderRadius: "12px",
        padding: "1.5rem",
        maxWidth: "1200px",
        margin: "0 auto 2rem auto",
        boxShadow: "0 8px 24px rgba(0,0,0,0.15)"
      }}>
        <h3 style={{
          color: "#333",
          marginBottom: "1rem",
          textAlign: "center",
          fontSize: "1.3rem"
        }}>
          ✨ Supported Model Types
        </h3>
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
          gap: "1rem",
          marginTop: "1rem"
        }}>
          {[
            { name: "Decision Tree", icon: "🌳", desc: "Classification & Regression" },
            { name: "Linear Regression", icon: "📈", desc: "1D & 2D visualizations" },
            { name: "Logistic Regression", icon: "📊", desc: "Binary classification" },
            { name: "SVM", icon: "🎯", desc: "Support Vector Machines" },
            { name: "KNN", icon: "🔍", desc: "K-Nearest Neighbors" },
            { name: "K-Means", icon: "🔵", desc: "Clustering algorithm" },
            { name: "Hierarchical", icon: "🌲", desc: "Agglomerative clustering" },
            { name: "Naive Bayes", icon: "🎲", desc: "Probabilistic classifier" }
          ].map((model, idx) => (
            <div key={idx} style={{
              backgroundColor: "#f8f9fa",
              padding: "1rem",
              borderRadius: "8px",
              border: "2px solid #e9ecef",
              textAlign: "center",
              transition: "transform 0.2s, box-shadow 0.2s",
              cursor: "default"
            }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "translateY(-4px)";
                e.currentTarget.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow = "none";
              }}>
              <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>{model.icon}</div>
              <div style={{ fontWeight: "600", color: "#333", marginBottom: "0.25rem" }}>
                {model.name}
              </div>
              <div style={{ fontSize: "0.85rem", color: "#6c757d" }}>
                {model.desc}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Upload Section */}
      <div
        style={{
          backgroundColor: "rgba(255, 255, 255, 0.98)",
          borderRadius: "12px",
          maxWidth: "1200px",
          margin: "0 auto",
          boxShadow: "0 8px 24px rgba(0,0,0,0.15)",
          overflow: "hidden"
        }}
      >
        <ModelUploader setExplanation={setExplanation} />
      </div>

      {/* Explanation Section */}
      {explanation && <ExplanationPanel explanation={explanation} />}

      {/* Footer */}
      <div style={{
        textAlign: "center",
        marginTop: "3rem",
        color: "#fff",
        textShadow: "1px 1px 2px rgba(0,0,0,0.5)"
      }}>
        <p style={{ fontSize: "0.9rem" }}>
          Built with ❤️ for Machine Learning enthusiasts
        </p>
      </div>

      {/* CSS Animations */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes rotate {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default App;
