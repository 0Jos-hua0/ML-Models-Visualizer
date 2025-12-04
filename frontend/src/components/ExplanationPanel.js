import React from 'react';

function ExplanationPanel({ explanation }) {
    if (!explanation) return null;

    const renderValue = (value) => {
        if (typeof value === 'object' && value !== null) {
            return <pre style={{
                margin: 0,
                whiteSpace: 'pre-wrap',
                fontSize: '0.9rem',
                backgroundColor: '#f8f9fa',
                padding: '0.5rem',
                borderRadius: '4px'
            }}>{JSON.stringify(value, null, 2)}</pre>;
        }
        return String(value);
    };

    const renderSection = (title, content, icon = "📌") => {
        if (!content) return null;

        return (
            <div style={{ marginBottom: "1.5rem" }}>
                <h4 style={{
                    color: "#007bff",
                    marginBottom: "0.75rem",
                    fontSize: "1.1rem",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem"
                }}>
                    <span>{icon}</span>
                    <span>{title}</span>
                </h4>
                <div style={{
                    backgroundColor: "#f8f9fa",
                    padding: "1rem",
                    borderRadius: "6px",
                    border: "1px solid #e9ecef"
                }}>
                    {typeof content === 'string' ? (
                        <p style={{ margin: 0, lineHeight: "1.6" }}>{content}</p>
                    ) : Array.isArray(content) ? (
                        <ol style={{ margin: 0, paddingLeft: "1.5rem" }}>
                            {content.map((item, idx) => (
                                <li key={idx} style={{ marginBottom: "0.5rem", lineHeight: "1.6" }}>{item}</li>
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
            <div style={{ marginBottom: "1.5rem" }}>
                <h4 style={{
                    color: "#007bff",
                    marginBottom: "0.75rem",
                    fontSize: "1.1rem",
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem"
                }}>
                    <span>⚙️</span>
                    <span>Model Parameters</span>
                </h4>
                <div style={{
                    backgroundColor: "#f8f9fa",
                    padding: "1rem",
                    borderRadius: "6px",
                    border: "1px solid #e9ecef"
                }}>
                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                        <tbody>
                            {Object.entries(params).map(([key, value]) => (
                                <tr key={key} style={{ borderBottom: "1px solid #dee2e6" }}>
                                    <td style={{
                                        padding: "0.5rem",
                                        fontWeight: "600",
                                        color: "#495057",
                                        width: "40%"
                                    }}>
                                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                    </td>
                                    <td style={{ padding: "0.5rem", color: "#212529" }}>
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
            padding: "2rem",
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
            maxWidth: "1000px",
            margin: "2rem auto"
        }}>
            <h3 style={{
                color: "#212529",
                marginBottom: "1.5rem",
                fontSize: "1.5rem",
                borderBottom: "2px solid #007bff",
                paddingBottom: "0.5rem"
            }}>
                🧠 Model Explanation
            </h3>

            {/* Model Type */}
            {explanation.model_type && (
                <div style={{
                    backgroundColor: "#007bff",
                    color: "white",
                    padding: "1rem",
                    borderRadius: "6px",
                    marginBottom: "1.5rem",
                    fontSize: "1.1rem",
                    fontWeight: "600"
                }}>
                    {explanation.model_type}
                </div>
            )}

            {/* Description */}
            {renderSection("What is this model?", explanation.description, "📖")}

            {/* Algorithm */}
            {explanation.algorithm && (
                <div style={{ marginBottom: "1.5rem" }}>
                    <h4 style={{
                        color: "#007bff",
                        marginBottom: "0.75rem",
                        fontSize: "1.1rem",
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem"
                    }}>
                        <span>🔬</span>
                        <span>Algorithm</span>
                    </h4>
                    <div style={{
                        backgroundColor: "#e7f3ff",
                        padding: "0.75rem 1rem",
                        borderRadius: "6px",
                        border: "1px solid #b3d9ff",
                        fontWeight: "500"
                    }}>
                        {explanation.algorithm}
                    </div>
                </div>
            )}

            {/* Equation */}
            {explanation.equation && (
                <div style={{ marginBottom: "1.5rem" }}>
                    <h4 style={{
                        color: "#007bff",
                        marginBottom: "0.75rem",
                        fontSize: "1.1rem",
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem"
                    }}>
                        <span>📐</span>
                        <span>Mathematical Equation</span>
                    </h4>
                    <div style={{
                        backgroundColor: "#fff3cd",
                        padding: "1rem",
                        borderRadius: "6px",
                        border: "1px solid #ffc107",
                        fontFamily: "monospace",
                        fontSize: "1.1rem",
                        textAlign: "center"
                    }}>
                        {explanation.equation}
                    </div>
                </div>
            )}

            {/* Parameters */}
            {explanation.parameters && renderParameters(explanation.parameters)}

            {/* How it works */}
            {renderSection("How does it work?", explanation.how_it_works, "⚡")}

            {/* Interpretation */}
            {renderSection("Interpretation", explanation.interpretation, "💡")}

            {/* Additional Info Sections */}
            {explanation.tree_info && renderSection("Tree Information", explanation.tree_info, "🌲")}
            {explanation.cluster_info && renderSection("Cluster Information", explanation.cluster_info, "🔵")}
            {explanation.support_info && renderSection("Support Vector Information", explanation.support_info, "🎯")}
            {explanation.cluster_distribution && renderSection("Cluster Distribution", explanation.cluster_distribution, "📊")}

            {/* Classes */}
            {explanation.classes && (
                <div style={{ marginBottom: "1.5rem" }}>
                    <h4 style={{
                        color: "#007bff",
                        marginBottom: "0.75rem",
                        fontSize: "1.1rem",
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem"
                    }}>
                        <span>🏷️</span>
                        <span>Classes ({explanation.n_classes || explanation.classes.length})</span>
                    </h4>
                    <div style={{
                        backgroundColor: "#d4edda",
                        padding: "0.75rem 1rem",
                        borderRadius: "6px",
                        border: "1px solid #c3e6cb"
                    }}>
                        {explanation.classes.join(', ')}
                    </div>
                </div>
            )}

            {/* Feature Importances */}
            {explanation.feature_importances && (
                <div style={{ marginBottom: "1.5rem" }}>
                    <h4 style={{
                        color: "#007bff",
                        marginBottom: "0.75rem",
                        fontSize: "1.1rem",
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem"
                    }}>
                        <span>⭐</span>
                        <span>Feature Importances</span>
                    </h4>
                    <div style={{
                        backgroundColor: "#f8f9fa",
                        padding: "1rem",
                        borderRadius: "6px",
                        border: "1px solid #e9ecef"
                    }}>
                        {explanation.feature_importances.map((importance, idx) => (
                            <div key={idx} style={{ marginBottom: "0.5rem" }}>
                                <div style={{
                                    display: "flex",
                                    justifyContent: "space-between",
                                    marginBottom: "0.25rem",
                                    fontSize: "0.9rem"
                                }}>
                                    <span>Feature {idx}</span>
                                    <span style={{ fontWeight: "600" }}>{(importance * 100).toFixed(2)}%</span>
                                </div>
                                <div style={{
                                    backgroundColor: "#e9ecef",
                                    borderRadius: "4px",
                                    height: "8px",
                                    overflow: "hidden"
                                }}>
                                    <div style={{
                                        backgroundColor: "#007bff",
                                        height: "100%",
                                        width: `${importance * 100}%`,
                                        transition: "width 0.3s ease"
                                    }}></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Variant Info (for Naive Bayes) */}
            {explanation.variant && (
                <div style={{ marginBottom: "1.5rem" }}>
                    <h4 style={{
                        color: "#007bff",
                        marginBottom: "0.75rem",
                        fontSize: "1.1rem",
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem"
                    }}>
                        <span>🔀</span>
                        <span>Model Variant</span>
                    </h4>
                    <div style={{
                        backgroundColor: "#fff3cd",
                        padding: "0.75rem 1rem",
                        borderRadius: "6px",
                        border: "1px solid #ffc107"
                    }}>
                        {explanation.variant}
                    </div>
                </div>
            )}

            {/* Linkage Types (for Hierarchical) */}
            {explanation.linkage_types && (
                <div style={{ marginBottom: "1.5rem" }}>
                    <h4 style={{
                        color: "#007bff",
                        marginBottom: "0.75rem",
                        fontSize: "1.1rem",
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem"
                    }}>
                        <span>🔗</span>
                        <span>Linkage Types</span>
                    </h4>
                    <div style={{
                        backgroundColor: "#f8f9fa",
                        padding: "1rem",
                        borderRadius: "6px",
                        border: "1px solid #e9ecef"
                    }}>
                        {Object.entries(explanation.linkage_types).map(([key, value]) => (
                            <div key={key} style={{ marginBottom: "0.5rem" }}>
                                <strong style={{ color: "#007bff" }}>{key}:</strong> {value}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default ExplanationPanel;
