import React, { useState } from "react";
import ModelUploader from "./components/ModelUploader";
import ExplanationPanel from "./components/ExplanationPanel";

// Import main assets
import logoIcon from "./assets/icon/stellarbasin.png";
import decisionTreeIcon from "./assets/icon/decision_tree.png";
import linearRegIcon from "./assets/icon/linear_regression.png";
import logisticRegIcon from "./assets/icon/logistic_regression.png";
import svmIcon from "./assets/icon/svm.png";
import knnIcon from "./assets/icon/knn.png";
import kmeansIcon from "./assets/icon/kmeans.png";
import hierarchicalIcon from "./assets/icon/hierarchical.png";
import nbIcon from "./assets/icon/naive_bayes.png";

function App() {
  const [explanation, setExplanation] = useState(null);

  return (
    <div
      style={{
        minHeight: "100vh",
        fontFamily: "'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
        background: "linear-gradient(135deg, #e0e4e7 0%, #c1bdb3 50%, #8fa6b0 100%)", // Gradient: Light Silver -> Silver -> Slate Tint
        padding: "0",
        margin: "0",
        color: "#000000" // Black text
      }}
    >
      {/* Top Navigation Bar */}
      <nav style={{
        backgroundColor: "#2e6171", // Blue Slate
        padding: "1rem 2rem",
        boxShadow: "0 4px 6px rgba(0,0,0,0.2)",
        borderBottom: "4px solid #a5402d" // Reddish Brown accent
      }}>
        <div style={{
          maxWidth: "1400px",
          margin: "0 auto",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between"
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            <div style={{
              width: "48px",
              height: "48px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: "white",
              borderRadius: "8px",
              padding: "4px"
            }}>
              <img
                src={logoIcon}
                alt="Logo"
                style={{ width: "100%", height: "100%", objectFit: "contain" }}
              />
            </div>
            <h1 style={{
              margin: 0,
              fontSize: "1.5rem",
              color: "#ffffff",
              fontWeight: "700",
              letterSpacing: "0.5px",
              textShadow: "1px 1px 2px rgba(0,0,0,0.3)"
            }}>
              StellarBasin
            </h1>
          </div>
          <div style={{
            fontSize: "0.95rem",
            color: "#c1bdb3", // Silver (light)
            fontWeight: "500"
          }}>
            ML Model Visualizer
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div style={{
        maxWidth: "1400px",
        margin: "0 auto",
        padding: "2rem 1rem"
      }}>
        {/* Hero Section */}
        <div style={{
          textAlign: "center",
          marginBottom: "3rem",
          padding: "2rem 1rem"
        }}>
          <h2 style={{
            fontSize: "2.5rem",
            color: "#000000",
            marginBottom: "1rem",
            fontWeight: "800"
          }}>
            Visualize & Understand Your Models
          </h2>
          <p style={{
            fontSize: "1.1rem",
            color: "#2e6171", // Blue Slate
            maxWidth: "700px",
            margin: "0 auto",
            lineHeight: "1.6",
            fontWeight: "600"
          }}>
            Upload your trained machine learning models and get instant visualizations
            with comprehensive explanations
          </p>
        </div>

        {/* Supported Models Grid */}
        <div style={{
          backgroundColor: "#ffffff", // White card
          borderRadius: "12px",
          padding: "2.5rem",
          marginBottom: "2.5rem",
          boxShadow: "0 8px 20px rgba(0,0,0,0.1)",
          border: "1px solid #2e6171" // Border
        }}>
          <h3 style={{
            color: "#2e6171", // Blue Slate
            marginBottom: "2rem",
            fontSize: "1.4rem",
            fontWeight: "700",
            textAlign: "center",
            textTransform: "uppercase",
            letterSpacing: "1px",
            borderBottom: "2px solid #a5402d", // Accent
            display: "inline-block",
            paddingBottom: "0.5rem",
            position: "relative",
            left: "50%",
            transform: "translateX(-50%)"
          }}>
            Supported Model Types
          </h3>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", // Reduced width for tighter packing
            gap: "1rem" // Reduced gap
          }}>
            {[
              { name: "Decision Tree", icon: decisionTreeIcon },
              { name: "Linear Regression", icon: linearRegIcon },
              { name: "Logistic Regression", icon: logisticRegIcon },
              { name: "SVM", icon: svmIcon },
              { name: "KNN", icon: knnIcon },
              { name: "K-Means", icon: kmeansIcon },
              { name: "Hierarchical", icon: hierarchicalIcon },
              { name: "Naive Bayes", icon: nbIcon }
            ].map((model, idx) => {
              return (
                <div key={idx} style={{
                  backgroundColor: "#c1bdb3", // Silver
                  padding: "0.8rem", // Reduced padding
                  borderRadius: "8px",
                  textAlign: "center",
                  border: "2px solid #2e6171", // Blue Slate border
                  transition: "all 0.3s ease",
                  cursor: "default",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "space-between", // Distribute space
                  minHeight: "160px" // Reduced height
                }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "translateY(-5px)";
                    e.currentTarget.style.boxShadow = "0 8px 16px rgba(46, 97, 113, 0.3)";
                    e.currentTarget.style.borderColor = "#a5402d"; // Highlight with Reddish Brown
                    e.currentTarget.style.backgroundColor = "#ffffff";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "translateY(0)";
                    e.currentTarget.style.boxShadow = "none";
                    e.currentTarget.style.borderColor = "#2e6171";
                    e.currentTarget.style.backgroundColor = "#c1bdb3";
                  }}>
                  <div style={{
                    marginBottom: "0.25rem", // Drastically reduced spacing
                    width: "125px", // Increased icon size
                    height: "125px",
                    display: "flex", // Ensure flex centering
                    alignItems: "center",
                    justifyContent: "center"
                  }}>
                    <img
                      src={model.icon}
                      alt={model.name}
                      style={{ width: "100%", height: "100%", objectFit: "contain" }}
                    />
                  </div>
                  <div style={{
                    fontWeight: "700",
                    color: "#000000",
                    fontSize: "0.9rem", // Slightly smaller text for compact fit
                    lineHeight: "1.1"
                  }}>
                    {model.name}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Upload Section */}
        <div style={{
          backgroundColor: "#ffffff",
          borderRadius: "12px",
          boxShadow: "0 8px 20px rgba(0,0,0,0.1)",
          overflow: "hidden",
          border: "1px solid #2e6171"
        }}>
          <div style={{
            backgroundColor: "#2e6171", // Blue Slate
            padding: "1.5rem 2rem",
            borderBottom: "4px solid #a5402d" // Accent
          }}>
            <h3 style={{
              margin: 0,
              color: "#ffffff",
              fontSize: "1.3rem",
              fontWeight: "700"
            }}>
              Upload Your Model
            </h3>
          </div>
          <ModelUploader setExplanation={setExplanation} />
        </div>

        {/* Explanation Section */}
        {explanation && (
          <div style={{ marginTop: "2.5rem" }}>
            <ExplanationPanel explanation={explanation} />
          </div>
        )}
      </div>

      {/* Footer */}
      <footer style={{
        backgroundColor: "#2e6171", // Blue Slate
        color: "#c1bdb3", // Silver
        textAlign: "center",
        padding: "2rem 1rem",
        marginTop: "4rem",
        borderTop: "4px solid #a5402d"
      }}>
        <p style={{ margin: 0, fontSize: "0.95rem", fontWeight: "500", color: "#ffffff" }}>
          StellarBasin • ML Model Visualization
        </p>
        <p style={{ margin: "0.5rem 0 0 0", fontSize: "0.85rem", color: "#c1bdb3" }}>
          Built with custom styling and precision
        </p>
      </footer>

      {/* Global Styles */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        * {
          box-sizing: border-box;
        }
        
        body {
          margin: 0;
          padding: 0;
        }
      `}</style>
    </div>
  );
}

export default App;
