import React from 'react';
import analyzeIcon from "../assets/icon/analyze.png";

function ExplanationPanel({ explanation }) {
    if (!explanation) return null;

    const renderValue = (value) => {
        if (typeof value === 'object' && value !== null) {
            return <pre style={{
                margin: 0,
                whiteSpace: 'pre-wrap',
                fontSize: '0.95rem',
                backgroundColor: '#ffffff',
                padding: '1rem',
                borderRadius: '4px',
                border: '1px solid #c1bdb3',
                color: '#000000',
                fontWeight: "500"
            }}>{JSON.stringify(value, null, 2)}</pre>;
        }
        return String(value);
    };

    const renderSection = (title, content, icon = "📌", headerColor = "#2e6171") => {
        if (!content) return null;

        return (
            <div style={{ marginBottom: "2rem" }}>
                <h4 style={{
                    color: headerColor,
                    marginBottom: "0.8rem",
                    fontSize: "1.2rem",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.7rem",
                    fontWeight: "700",
                    borderBottom: "1px solid #c1bdb3",
                    paddingBottom: "0.5rem"
                }}>
                    <span>{icon}</span>
                    <span>{title}</span>
                </h4>
                <div style={{
                    backgroundColor: "#f4f4f4", // Slightly lighter than silver for readability
                    padding: "1.5rem",
                    borderRadius: "4px",
                    borderLeft: `4px solid ${headerColor}`,
                    boxShadow: "0 2px 4px rgba(0,0,0,0.05)"
                }}>
                    {typeof content === 'string' ? (
                        <p style={{ margin: 0, lineHeight: "1.7", color: "#000000", fontSize: "1rem" }}>{content}</p>
                    ) : Array.isArray(content) ? (
                        <ol style={{ margin: 0, paddingLeft: "1.5rem", color: "#000000" }}>
                            {content.map((item, idx) => (
                                <li key={idx} style={{ marginBottom: "0.5rem", lineHeight: "1.7", fontSize: "1rem" }}>{item}</li>
                            ))}
                        </ol>
                    ) : (
                        <div>{renderValue(content)}</div>
                    )}
                </div>
            </div>
        );
    };

    const renderParameters = (params) => {
        if (!params || typeof params !== 'object') return null;

        return (
            <div style={{ marginBottom: "2rem" }}>
                <h4 style={{
                    color: "#2e6171",
                    marginBottom: "0.8rem",
                    fontSize: "1.2rem",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.7rem",
                    fontWeight: "700",
                    borderBottom: "1px solid #c1bdb3",
                    paddingBottom: "0.5rem"
                }}>
                    <span>⚙️</span>
                    <span>Model Parameters</span>
                </h4>
                <div style={{
                    backgroundColor: "#ffffff",
                    padding: "0",
                    borderRadius: "4px",
                    border: "1px solid #c1bdb3",
                    overflow: "hidden"
                }}>
                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                        <tbody>
                            {Object.entries(params).map(([key, value], idx) => (
                                <tr key={key} style={{
                                    borderBottom: idx < Object.entries(params).length - 1 ? "1px solid #e0e0e0" : "none",
                                    backgroundColor: idx % 2 === 0 ? "#f9f9f9" : "#ffffff"
                                }}>
                                    <td style={{
                                        padding: "0.8rem 1.2rem",
                                        fontWeight: "600",
                                        color: "#2e6171",
                                        width: "40%",
                                        fontSize: "0.95rem"
                                    }}>
                                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                    </td>
                                    <td style={{ padding: "0.8rem 1.2rem", color: "#000000", fontSize: "0.95rem" }}>
                                        {renderValue(value)}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        );
    };

    return (
        <div style={{
            backgroundColor: "white",
            borderRadius: "12px",
            boxShadow: "0 8px 20px rgba(0,0,0,0.1)",
            border: "1px solid #2e6171",
            overflow: "hidden"
        }}>
            {/* Header */}
            <div style={{
                backgroundColor: "#2e6171", // Blue Slate
                padding: "1.5rem 2rem",
                borderBottom: "4px solid #a5402d" // Accent
            }}>
                <div style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "1rem"
                }}>
                    <div style={{ width: "40px", height: "40px", backgroundColor: "white", borderRadius: "50%", padding: "5px", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <img src={analyzeIcon} alt="Explanation" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
                    </div>
                    <h3 style={{
                        margin: 0,
                        color: "white",
                        fontSize: "1.4rem",
                        fontWeight: "700"
                    }}>
                        Model Explanation
                    </h3>
                </div>
            </div>

            <div style={{ padding: "2.5rem" }}>
                {/* Model Type Badge */}
                {explanation.model_type && (
                    <div style={{
                        backgroundColor: "#2e6171",
                        color: "white",
                        padding: "1rem 2rem",
                        borderRadius: "4px",
                        marginBottom: "2.5rem",
                        fontSize: "1.2rem",
                        fontWeight: "700",
                        textAlign: "center",
                        boxShadow: "0 4px 10px rgba(46, 97, 113, 0.3)",
                        border: "1px solid #1a3c48"
                    }}>
                        {explanation.model_type}
                    </div>
                )}

                {/* Description */}
                {renderSection("What is this model?", explanation.description, "📖", "#2e6171")}

                {/* Algorithm */}
                {explanation.algorithm && (
                    <div style={{ marginBottom: "2rem" }}>
                        <h4 style={{
                            color: "#2e6171",
                            marginBottom: "0.8rem",
                            fontSize: "1.2rem",
                            display: "flex",
                            alignItems: "center",
                            gap: "0.7rem",
                            fontWeight: "700",
                            borderBottom: "1px solid #c1bdb3",
                            paddingBottom: "0.5rem"
                        }}>
                            <span>🔬</span>
                            <span>Algorithm</span>
                        </h4>
                        <div style={{
                            backgroundColor: "#c1bdb3", // Silver bg
                            padding: "1.2rem 1.5rem",
                            borderRadius: "4px",
                            border: "2px solid #2e6171",
                            fontWeight: "700",
                            color: "#000000",
                            fontSize: "1.1rem"
                        }}>
                            {explanation.algorithm}
                        </div>
                    </div>
                )}

                {/* Equation */}
                {explanation.equation && (
                    <div style={{ marginBottom: "2rem" }}>
                        <h4 style={{
                            color: "#2e6171",
                            marginBottom: "0.8rem",
                            fontSize: "1.2rem",
                            display: "flex",
                            alignItems: "center",
                            gap: "0.7rem",
                            fontWeight: "700",
                            borderBottom: "1px solid #c1bdb3",
                            paddingBottom: "0.5rem"
                        }}>
                            <span>📐</span>
                            <span>Mathematical Equation</span>
                        </h4>
                        <div style={{
                            backgroundColor: "#222222", // Dark background for equation
                            padding: "1.5rem",
                            borderRadius: "4px",
                            border: "2px solid #a5402d", // Accent border
                            fontFamily: "'Courier New', monospace",
                            fontSize: "1.1rem",
                            textAlign: "center",
                            color: "#ffffff",
                            fontWeight: "600",
                            letterSpacing: "0.5px"
                        }}>
                            {explanation.equation}
                        </div>
                    </div>
                )}

                {/* Parameters */}
                {explanation.parameters && renderParameters(explanation.parameters)}

                {/* How it works */}
                {renderSection("How does it work?", explanation.how_it_works, "⚡", "#a5402d")}

                {/* Interpretation */}
                {renderSection("Interpretation", explanation.interpretation, "💡", "#2e6171")}

                {/* Additional Info Sections */}
                {explanation.tree_info && renderSection("Tree Information", explanation.tree_info, "🌲", "#2e6171")}
                {explanation.cluster_info && renderSection("Cluster Information", explanation.cluster_info, "🔵", "#2e6171")}
                {explanation.support_info && renderSection("Support Vector Information", explanation.support_info, "🎯", "#2e6171")}
                {explanation.cluster_distribution && renderSection("Cluster Distribution", explanation.cluster_distribution, "📊", "#2e6171")}

                {/* Classes */}
                {explanation.classes && (
                    <div style={{ marginBottom: "2rem" }}>
                        <h4 style={{
                            color: "#2e6171",
                            marginBottom: "0.8rem",
                            fontSize: "1.2rem",
                            display: "flex",
                            alignItems: "center",
                            gap: "0.7rem",
                            fontWeight: "700",
                            borderBottom: "1px solid #c1bdb3",
                            paddingBottom: "0.5rem"
                        }}>
                            <span>🏷️</span>
                            <span>Classes ({explanation.n_classes || explanation.classes.length})</span>
                        </h4>
                        <div style={{
                            backgroundColor: "#e0e0e0",
                            padding: "1.2rem 1.5rem",
                            borderRadius: "4px",
                            border: "2px solid #2e6171",
                            color: "#000000",
                            fontWeight: "600",
                            fontSize: "1rem"
                        }}>
                            {explanation.classes.join(', ')}
                        </div>
                    </div>
                )}

                {/* Feature Importances */}
                {explanation.feature_importances && (
                    <div style={{ marginBottom: "2rem" }}>
                        <h4 style={{
                            color: "#2e6171",
                            marginBottom: "0.8rem",
                            fontSize: "1.2rem",
                            display: "flex",
                            alignItems: "center",
                            gap: "0.7rem",
                            fontWeight: "700",
                            borderBottom: "1px solid #c1bdb3",
                            paddingBottom: "0.5rem"
                        }}>
                            <span>⭐</span>
                            <span>Feature Importances</span>
                        </h4>
                        <div style={{
                            backgroundColor: "#f4f4f4",
                            padding: "1.5rem",
                            borderRadius: "4px",
                            border: "1px solid #c1bdb3"
                        }}>
                            {explanation.feature_importances.map((importance, idx) => (
                                <div key={idx} style={{ marginBottom: "1.2rem" }}>
                                    <div style={{
                                        display: "flex",
                                        justifyContent: "space-between",
                                        marginBottom: "0.5rem",
                                        fontSize: "0.95rem",
                                        color: "#000000",
                                        fontWeight: "600"
                                    }}>
                                        <span>Feature {idx}</span>
                                        <span>{(importance * 100).toFixed(2)}%</span>
                                    </div>
                                    <div style={{
                                        backgroundColor: "#c1bdb3",
                                        borderRadius: "4px",
                                        height: "12px",
                                        overflow: "hidden"
                                    }}>
                                        <div style={{
                                            backgroundColor: "#2e6171",
                                            height: "100%",
                                            width: `${importance * 100}%`,
                                            transition: "width 0.5s ease",
                                            borderRadius: "4px"
                                        }}></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Variant Info */}
                {explanation.variant && (
                    <div style={{ marginBottom: "2rem" }}>
                        <h4 style={{
                            color: "#2e6171",
                            marginBottom: "0.8rem",
                            fontSize: "1.2rem",
                            display: "flex",
                            alignItems: "center",
                            gap: "0.7rem",
                            fontWeight: "700",
                            borderBottom: "1px solid #c1bdb3",
                            paddingBottom: "0.5rem"
                        }}>
                            <span>🔀</span>
                            <span>Model Variant</span>
                        </h4>
                        <div style={{
                            backgroundColor: "#f4f4f4",
                            padding: "1.2rem 1.5rem",
                            borderRadius: "4px",
                            border: "2px solid #a5402d",
                            color: "#a5402d",
                            fontWeight: "700",
                            fontSize: "1rem"
                        }}>
                            {explanation.variant}
                        </div>
                    </div>
                )}

                {/* Linkage Types */}
                {explanation.linkage_types && (
                    <div style={{ marginBottom: "2rem" }}>
                        <h4 style={{
                            color: "#2e6171",
                            marginBottom: "0.8rem",
                            fontSize: "1.2rem",
                            display: "flex",
                            alignItems: "center",
                            gap: "0.7rem",
                            fontWeight: "700",
                            borderBottom: "1px solid #c1bdb3",
                            paddingBottom: "0.5rem"
                        }}>
                            <span>🔗</span>
                            <span>Linkage Types</span>
                        </h4>
                        <div style={{
                            backgroundColor: "#f4f4f4",
                            padding: "1.5rem",
                            borderRadius: "4px",
                            border: "1px solid #c1bdb3"
                        }}>
                            {Object.entries(explanation.linkage_types).map(([key, value]) => (
                                <div key={key} style={{ marginBottom: "0.8rem", color: "#000000" }}>
                                    <strong style={{ color: "#2e6171" }}>{key}:</strong> {value}
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default ExplanationPanel;
