import joblib
import numpy as np
import io
import base64
import plotly.express as px
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
import json
from sklearn.exceptions import InconsistentVersionWarning
import matplotlib
import pandas as pd
from sklearn.metrics import classification_report


matplotlib.use('Agg')  # Use non-GUI backend for matplotlib


def load_model_object(file_path):
    return joblib.load(file_path)

def load_model(file_path):
    try:
        # Suppress the warning about scikit-learn version mismatch
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            model = joblib.load(file_path)

        error_message, fig_or_base64, report = detect_errors_and_visualize(model)

        if error_message:
            return error_message, None

        if isinstance(fig_or_base64, go.Figure):
            fig_json = fig_or_base64.to_dict()
            return None, {'plotly': fig_json, 'report': report}

        return None, {'image': fig_or_base64, 'report': report}

    except Exception as e:
        return str(e), None

def detect_errors_and_visualize(model):
    try:
        if isinstance(model, DecisionTreeClassifier):
            fig = visualize_decision_tree(model)
            report = generate_report(model, "Decision Tree")
            return None, fig, report  # base64 image

        elif isinstance(model, KNeighborsClassifier):
            fig = visualize_knn(model)
            report = generate_report(model, "KNN")
            return None, fig, report  # raw go.Figure

        elif isinstance(model, LinearRegression):
            fig = visualize_linear_regression(model)
            report = generate_report(model, "Linear Regression")
            return None, fig, report

        elif isinstance(model, LogisticRegression):
            fig = visualize_logistic_regression(model)
            report = generate_report(model, "Logistic Regression")
            return None, fig, report

        else:
            return "Unsupported model type!", None, None

    except Exception as e:
        import traceback
        return f"Visualization error: {str(e)}\n{traceback.format_exc()}", None, None

    
def visualize_decision_tree(model):
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model, filled=True, rounded=True, class_names=True, ax=ax)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # Return base64 string
    return encoded_img


import plotly.graph_objects as go
import numpy as np

def visualize_knn(model):
    # Get class distribution
    unique_classes, counts = np.unique(model.classes_, return_counts=True)

    # Bar chart for class counts
    bar = go.Bar(
        x=unique_classes,
        y=counts,
        text=[f"{c} samples" for c in counts],
        textposition="auto",
        marker=dict(color="cyan")
    )

    # Layout with model metadata annotations
    layout = go.Layout(
        title="KNN Model Summary",
        xaxis=dict(title="Class Labels"),
        yaxis=dict(title="Number of Samples"),
        template="plotly_dark",
        annotations=[
            dict(
                text=f"<b>n_neighbors</b>: {model.n_neighbors}<br>"
                     f"<b>Weights</b>: {model.weights}<br>"
                     f"<b>Algorithm</b>: {model.algorithm}",
                align="left",
                showarrow=False,
                xref="paper", yref="paper",
                x=1.05, y=1,
                bordercolor="gray",
                borderwidth=1,
                bgcolor="black",
                font=dict(color="white", size=12)
            )
        ],
        margin=dict(r=150)  # Room for metadata annotation
    )

    fig = go.Figure(data=[bar], layout=layout)
    return fig






def visualize_linear_regression(model):
    fig = go.Figure()

    try:
        if model.coef_.shape[0] == 2:
            # Generate a grid for two features
            x_range = np.linspace(800, 3000, 10)
            y_range = np.linspace(1, 5, 10)
            x_grid, y_grid = np.meshgrid(x_range, y_range)

            # Handle models without feature_names_in_
            if hasattr(model, "feature_names_in_"):
                col_names = model.feature_names_in_
            else:
                col_names = [f"Feature_{i}" for i in range(2)]

            grid_df = pd.DataFrame(
                np.c_[x_grid.ravel(), y_grid.ravel()],
                columns=col_names
            )

            price_grid = model.predict(grid_df).reshape(x_grid.shape)
            fig.add_trace(go.Surface(z=price_grid, x=x_grid, y=y_grid))
            fig.update_layout(
                title="Linear Regression Surface",
                scene=dict(
                    xaxis_title="Feature 1",
                    yaxis_title="Feature 2",
                    zaxis_title="Prediction"
                ),
                template="plotly_dark"
            )
        else:
            # Plot coefficients for 1D or >2D
            coefficients = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
            fig.add_trace(go.Bar(x=list(range(len(coefficients))), y=coefficients))
            fig.update_layout(
                title="Linear Regression Coefficients",
                xaxis_title="Feature Index",
                yaxis_title="Coefficient",
                template="plotly_dark"
            )
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error during visualization: {e}",
            showarrow=False,
            font=dict(color="red", size=14),
            xref="paper", yref="paper", x=0.5, y=0.5
        )
        fig.update_layout(template="plotly_dark")

    return fig.to_dict()


def visualize_logistic_regression(model):
    coefficients = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
    fig = go.Figure([go.Bar(x=list(range(len(coefficients))), y=coefficients)])
    fig.update_layout(
        title="Logistic Regression Coefficients",
        xaxis_title="Feature Index",
        yaxis_title="Coefficient",
        template="plotly_dark"
    )
    return fig












def generate_report(model, model_type, X=None, y=None):
    report = {
        "model_type": model_type,
        "parameters": model.get_params(),
    }
    try:
        if hasattr(model, 'score') and X is not None and y is not None:
            report["score"] = model.score(X, y)
        else:
            report["score"] = "N/A"

        if hasattr(model, 'classes_'):
            report["classes"] = model.classes_.tolist()

        if hasattr(model, 'coef_'):
            report["coefficients"] = model.coef_.tolist()

        if hasattr(model, 'feature_importances_'):
            report["feature_importances"] = model.feature_importances_.tolist()

        # ✅ Add this: full classification report
        if X is not None and y is not None and hasattr(model, "predict"):
            y_pred = model.predict(X)
            report["classification_report"] = classification_report(y, y_pred, output_dict=True)

    except Exception as e:
        report["note"] = f"Some report data could not be extracted: {e}"

    return report
