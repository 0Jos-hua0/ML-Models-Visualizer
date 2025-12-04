import joblib
import numpy as np
import io
import base64
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.cluster import KMeans, AgglomerativeClustering
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

# Use non-GUI backend for matplotlib
import matplotlib
matplotlib.use('Agg')


def convert_to_serializable(obj):
    """Convert NumPy types to JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def load_model(file_path):
    """Load a model from file and generate visualization + explanation."""
    try:
        # Suppress scikit-learn version warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            model = joblib.load(file_path)

        # Detect model type and generate visualization + explanation
        error_message, visualization, explanation = detect_and_visualize(model)

        if error_message:
            return error_message, None

        # Return the results (convert all NumPy types to JSON-serializable types)
        result = {
            'visualization': visualization,
            'explanation': explanation,
            'model_type': explanation.get('model_type', 'Unknown')
        }

        return None, convert_to_serializable(result)

    except Exception as e:
        return f"Error loading model: {str(e)}", None


def detect_and_visualize(model):
    """Detect model type and generate appropriate visualization and explanation."""
    try:
        # Decision Tree
        if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            viz = visualize_decision_tree(model)
            exp = explain_decision_tree(model)
            return None, viz, exp

        # Linear Regression
        elif isinstance(model, LinearRegression):
            viz = visualize_linear_regression(model)
            exp = explain_linear_regression(model)
            return None, viz, exp

        # Logistic Regression
        elif isinstance(model, LogisticRegression):
            viz = visualize_logistic_regression(model)
            exp = explain_logistic_regression(model)
            return None, viz, exp

        # SVM (Support Vector Machine)
        elif isinstance(model, (SVC, SVR)):
            viz = visualize_svm(model)
            exp = explain_svm(model)
            return None, viz, exp

        # KNN
        elif isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
            viz = visualize_knn(model)
            exp = explain_knn(model)
            return None, viz, exp

        # K-Means Clustering
        elif isinstance(model, KMeans):
            viz = visualize_kmeans(model)
            exp = explain_kmeans(model)
            return None, viz, exp

        # Hierarchical Clustering
        elif isinstance(model, AgglomerativeClustering):
            viz = visualize_hierarchical(model)
            exp = explain_hierarchical(model)
            return None, viz, exp

        # Naive Bayes
        elif isinstance(model, (GaussianNB, MultinomialNB, BernoulliNB)):
            viz = visualize_naive_bayes(model)
            exp = explain_naive_bayes(model)
            return None, viz, exp

        else:
            return f"Unsupported model type: {type(model).__name__}", None, None

    except Exception as e:
        import traceback
        return f"Visualization error: {str(e)}\n{traceback.format_exc()}", None, None


# ==================== DECISION TREE ====================

def visualize_decision_tree(model):
    """Generate Decision Tree visualization as base64 image."""
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model, filled=True, rounded=True, class_names=True, 
              feature_names=getattr(model, 'feature_names_in_', None), ax=ax)
    plt.title("Decision Tree Structure", fontsize=16, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return {'type': 'image', 'data': encoded_img}


def explain_decision_tree(model):
    """Generate explanation for Decision Tree."""
    is_classifier = isinstance(model, DecisionTreeClassifier)
    
    explanation = {
        'model_type': 'Decision Tree Classifier' if is_classifier else 'Decision Tree Regressor',
        'algorithm': 'Decision Tree',
        'description': 'A tree-based model that makes predictions by learning simple decision rules from data features.',
        'parameters': {
            'max_depth': model.max_depth or 'Unlimited',
            'min_samples_split': model.min_samples_split,
            'min_samples_leaf': model.min_samples_leaf,
            'criterion': model.criterion,
        },
        'tree_info': {
            'number_of_nodes': model.tree_.node_count,
            'number_of_leaves': model.tree_.n_leaves,
            'max_depth_actual': model.tree_.max_depth,
        },
        'how_it_works': [
            '1. Starts at the root node with all training data',
            '2. Splits data based on feature values that best separate classes/values',
            '3. Recursively creates branches until stopping criteria are met',
            '4. Makes predictions by traversing from root to leaf nodes'
        ]
    }

    if is_classifier and hasattr(model, 'classes_'):
        explanation['classes'] = model.classes_.tolist()
        explanation['n_classes'] = len(model.classes_)

    if hasattr(model, 'feature_importances_'):
        explanation['feature_importances'] = model.feature_importances_.tolist()

    return explanation


# ==================== LINEAR REGRESSION ====================

def visualize_linear_regression(model):
    """Generate Linear Regression visualization."""
    n_features = model.coef_.shape[0] if model.coef_.ndim == 1 else model.coef_.shape[1]

    if n_features == 1:
        # 1D: Simple line plot
        x_range = np.linspace(-10, 10, 100).reshape(-1, 1)
        y_pred = model.predict(x_range)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_pred, mode='lines', 
                                 name='Regression Line', line=dict(color='cyan', width=3)))
        fig.update_layout(
            title='Linear Regression: y = mx + b',
            xaxis_title='Feature (X)',
            yaxis_title='Prediction (y)',
            template='plotly_dark',
            annotations=[{
                'text': f'<b>Slope (m):</b> {model.coef_[0]:.4f}<br><b>Intercept (b):</b> {model.intercept_:.4f}',
                'showarrow': False,
                'xref': 'paper', 'yref': 'paper',
                'x': 0.02, 'y': 0.98,
                'bgcolor': 'rgba(0,0,0,0.7)',
                'font': {'color': 'white', 'size': 12}
            }]
        )

    elif n_features == 2:
        # 2D: 3D surface plot
        x_range = np.linspace(-10, 10, 20)
        y_range = np.linspace(-10, 10, 20)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        
        X_grid = np.c_[x_grid.ravel(), y_grid.ravel()]
        z_grid = model.predict(X_grid).reshape(x_grid.shape)

        fig = go.Figure(data=[go.Surface(z=z_grid, x=x_grid, y=y_grid, colorscale='Viridis')])
        fig.update_layout(
            title='Linear Regression Surface (2 Features)',
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Prediction'
            ),
            template='plotly_dark'
        )

    else:
        # Multi-dimensional: Show coefficients
        coefficients = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        fig = go.Figure(data=[go.Bar(
            x=[f'Feature {i}' for i in range(len(coefficients))],
            y=coefficients,
            marker=dict(color=coefficients, colorscale='RdBu', showscale=True)
        )])
        fig.update_layout(
            title=f'Linear Regression Coefficients ({n_features} Features)',
            xaxis_title='Features',
            yaxis_title='Coefficient Value',
            template='plotly_dark'
        )

    return {'type': 'plotly', 'data': fig.to_dict()}


def explain_linear_regression(model):
    """Generate explanation for Linear Regression."""
    coefficients = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
    
    return {
        'model_type': 'Linear Regression',
        'algorithm': 'Ordinary Least Squares (OLS)',
        'description': 'Predicts continuous values by fitting a linear equation to observed data.',
        'equation': 'y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ',
        'parameters': {
            'intercept': float(model.intercept_) if np.isscalar(model.intercept_) else model.intercept_.tolist(),
            'coefficients': coefficients.tolist(),
            'n_features': len(coefficients)
        },
        'how_it_works': [
            '1. Assumes a linear relationship between input features and output',
            '2. Finds the best-fit line that minimizes squared errors',
            '3. Uses coefficients to weight each feature\'s contribution',
            '4. Predicts by computing weighted sum of features plus intercept'
        ],
        'interpretation': 'Each coefficient shows how much the prediction changes when that feature increases by 1 unit.'
    }


# ==================== LOGISTIC REGRESSION ====================

def visualize_logistic_regression(model):
    """Generate Logistic Regression visualization."""
    n_features = model.coef_.shape[1] if model.coef_.ndim > 1 else len(model.coef_)

    if n_features == 1:
        # 1D: Sigmoid curve
        x_range = np.linspace(-10, 10, 200).reshape(-1, 1)
        y_prob = model.predict_proba(x_range)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(x_range)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_prob, mode='lines',
                                 name='Probability', line=dict(color='orange', width=3)))
        fig.add_hline(y=0.5, line_dash='dash', line_color='red', annotation_text='Decision Boundary (0.5)')
        fig.update_layout(
            title='Logistic Regression: Sigmoid Function',
            xaxis_title='Feature (X)',
            yaxis_title='Probability of Class 1',
            template='plotly_dark'
        )

    elif n_features == 2:
        # 2D: Decision boundary (contour plot)
        x_range = np.linspace(-10, 10, 100)
        y_range = np.linspace(-10, 10, 100)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        
        X_grid = np.c_[x_grid.ravel(), y_grid.ravel()]
        z_grid = model.predict_proba(X_grid)[:, 1].reshape(x_grid.shape) if hasattr(model, 'predict_proba') else model.decision_function(X_grid).reshape(x_grid.shape)

        fig = go.Figure(data=[go.Contour(z=z_grid, x=x_range, y=y_range, 
                                         colorscale='RdYlBu', showscale=True)])
        fig.update_layout(
            title='Logistic Regression Decision Boundary (2 Features)',
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            template='plotly_dark'
        )

    else:
        # Multi-dimensional: Show coefficients
        coefficients = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        fig = go.Figure(data=[go.Bar(
            x=[f'Feature {i}' for i in range(len(coefficients))],
            y=coefficients,
            marker=dict(color=coefficients, colorscale='Plasma', showscale=True)
        )])
        fig.update_layout(
            title=f'Logistic Regression Coefficients ({n_features} Features)',
            xaxis_title='Features',
            yaxis_title='Coefficient Value',
            template='plotly_dark'
        )

    return {'type': 'plotly', 'data': fig.to_dict()}


def explain_logistic_regression(model):
    """Generate explanation for Logistic Regression."""
    coefficients = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
    
    explanation = {
        'model_type': 'Logistic Regression',
        'algorithm': 'Logistic Regression (Maximum Likelihood)',
        'description': 'Predicts probability of binary outcomes using a sigmoid function.',
        'equation': 'P(y=1) = 1 / (1 + e^-(β₀ + β₁x₁ + ... + βₙxₙ))',
        'parameters': {
            'intercept': float(model.intercept_[0]) if hasattr(model.intercept_, '__iter__') else float(model.intercept_),
            'coefficients': coefficients.tolist(),
            'n_features': len(coefficients),
            'solver': model.solver,
            'penalty': model.penalty
        },
        'how_it_works': [
            '1. Computes a linear combination of input features',
            '2. Applies sigmoid function to map values to [0, 1] probability range',
            '3. Uses 0.5 as default threshold for binary classification',
            '4. Optimizes coefficients using maximum likelihood estimation'
        ]
    }

    if hasattr(model, 'classes_'):
        explanation['classes'] = model.classes_.tolist()

    return explanation


# ==================== SVM (Support Vector Machine) ====================

def visualize_svm(model):
    """Generate SVM visualization."""
    is_classifier = isinstance(model, SVC)
    
    # For SVM, we'll show support vectors and decision boundary info
    fig = go.Figure()
    
    if hasattr(model, 'support_vectors_'):
        n_support = len(model.support_vectors_)
        n_features = model.support_vectors_.shape[1]
        
        if n_features == 2:
            # 2D visualization of support vectors
            sv = model.support_vectors_
            fig.add_trace(go.Scatter(
                x=sv[:, 0], y=sv[:, 1],
                mode='markers',
                marker=dict(size=12, color='red', symbol='x', line=dict(width=2)),
                name='Support Vectors'
            ))
            fig.update_layout(
                title=f'SVM Support Vectors ({n_support} vectors)',
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                template='plotly_dark'
            )
        else:
            # Show support vector distribution
            fig.add_trace(go.Bar(
                x=[f'Class {i}' for i in range(len(model.n_support_))],
                y=model.n_support_,
                marker=dict(color='cyan')
            ))
            fig.update_layout(
                title='Number of Support Vectors per Class',
                xaxis_title='Class',
                yaxis_title='Count',
                template='plotly_dark'
            )
    else:
        # Fallback: Show kernel info
        fig.add_annotation(
            text=f'<b>SVM Model</b><br>Kernel: {model.kernel}<br>Support vectors information not available',
            showarrow=False,
            xref='paper', yref='paper',
            x=0.5, y=0.5,
            font=dict(size=16, color='white')
        )
        fig.update_layout(template='plotly_dark')

    return {'type': 'plotly', 'data': fig.to_dict()}


def explain_svm(model):
    """Generate explanation for SVM."""
    is_classifier = isinstance(model, SVC)
    
    explanation = {
        'model_type': 'Support Vector Machine (SVM) - ' + ('Classifier' if is_classifier else 'Regressor'),
        'algorithm': 'Support Vector Machine',
        'description': 'Finds optimal hyperplane that maximizes margin between classes or fits data with minimal error.',
        'parameters': {
            'kernel': model.kernel,
            'C': model.C,
            'gamma': model.gamma if hasattr(model, 'gamma') else 'N/A',
        },
        'how_it_works': [
            '1. Maps data to higher-dimensional space using kernel function',
            '2. Finds hyperplane that maximally separates classes (or fits data)',
            '3. Uses support vectors (critical data points) to define decision boundary',
            '4. Margin maximization ensures good generalization'
        ]
    }

    if hasattr(model, 'support_vectors_'):
        explanation['support_info'] = {
            'n_support_vectors': len(model.support_vectors_),
            'n_support_per_class': model.n_support_.tolist() if hasattr(model, 'n_support_') else 'N/A'
        }

    if is_classifier and hasattr(model, 'classes_'):
        explanation['classes'] = model.classes_.tolist()

    return explanation


# ==================== KNN ====================

def visualize_knn(model):
    """Generate KNN visualization."""
    if hasattr(model, 'classes_'):
        # Classifier: Show class distribution
        unique_classes, counts = np.unique(model.classes_, return_counts=True)
        
        fig = go.Figure(data=[go.Bar(
            x=unique_classes,
            y=counts,
            text=[f'{c} samples' for c in counts],
            textposition='auto',
            marker=dict(color='cyan')
        )])
        
        fig.update_layout(
            title=f'KNN Model - Class Distribution (k={model.n_neighbors})',
            xaxis_title='Class Labels',
            yaxis_title='Number of Samples',
            template='plotly_dark',
            annotations=[{
                'text': f'<b>Algorithm:</b> {model.algorithm}<br><b>Weights:</b> {model.weights}<br><b>Metric:</b> {model.metric}',
                'showarrow': False,
                'xref': 'paper', 'yref': 'paper',
                'x': 0.98, 'y': 0.98,
                'xanchor': 'right',
                'bgcolor': 'rgba(0,0,0,0.7)',
                'font': {'color': 'white', 'size': 12}
            }]
        )
    else:
        # Regressor: Show parameter info
        fig = go.Figure()
        fig.add_annotation(
            text=f'<b>KNN Regressor</b><br>k = {model.n_neighbors}<br>Algorithm: {model.algorithm}<br>Weights: {model.weights}',
            showarrow=False,
            xref='paper', yref='paper',
            x=0.5, y=0.5,
            font=dict(size=16, color='white')
        )
        fig.update_layout(template='plotly_dark', title='KNN Regressor Configuration')

    return {'type': 'plotly', 'data': fig.to_dict()}


def explain_knn(model):
    """Generate explanation for KNN."""
    is_classifier = isinstance(model, KNeighborsClassifier)
    
    explanation = {
        'model_type': 'K-Nearest Neighbors (KNN) - ' + ('Classifier' if is_classifier else 'Regressor'),
        'algorithm': 'K-Nearest Neighbors',
        'description': 'Makes predictions based on the k closest training examples in feature space.',
        'parameters': {
            'n_neighbors': model.n_neighbors,
            'weights': model.weights,
            'algorithm': model.algorithm,
            'metric': model.metric,
            'p': model.p
        },
        'how_it_works': [
            '1. Stores all training data (lazy learning)',
            '2. For new data point, finds k nearest neighbors using distance metric',
            '3. For classification: Uses majority vote of k neighbors',
            '4. For regression: Uses average of k neighbors\' values'
        ],
        'interpretation': f'Predictions are based on the {model.n_neighbors} most similar training examples.'
    }

    if is_classifier and hasattr(model, 'classes_'):
        explanation['classes'] = model.classes_.tolist()
        explanation['n_classes'] = len(model.classes_)

    return explanation


# ==================== K-MEANS CLUSTERING ====================

def visualize_kmeans(model):
    """Generate K-Means visualization."""
    n_clusters = model.n_clusters
    centers = model.cluster_centers_
    n_features = centers.shape[1]

    if n_features == 2:
        # 2D scatter of cluster centers
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1],
            mode='markers+text',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
            text=[f'C{i}' for i in range(n_clusters)],
            textposition='top center',
            name='Cluster Centers'
        ))
        fig.update_layout(
            title=f'K-Means Cluster Centers (k={n_clusters})',
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            template='plotly_dark'
        )
    elif n_features == 3:
        # 3D scatter of cluster centers
        fig = go.Figure(data=[go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='diamond'),
            text=[f'Cluster {i}' for i in range(n_clusters)],
            name='Cluster Centers'
        )])
        fig.update_layout(
            title=f'K-Means Cluster Centers (k={n_clusters})',
            scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Feature 3'),
            template='plotly_dark'
        )
    else:
        # Heatmap of cluster centers
        fig = go.Figure(data=go.Heatmap(
            z=centers,
            x=[f'Feature {i}' for i in range(n_features)],
            y=[f'Cluster {i}' for i in range(n_clusters)],
            colorscale='Viridis'
        ))
        fig.update_layout(
            title=f'K-Means Cluster Centers Heatmap ({n_clusters} clusters, {n_features} features)',
            xaxis_title='Features',
            yaxis_title='Clusters',
            template='plotly_dark'
        )

    return {'type': 'plotly', 'data': fig.to_dict()}


def explain_kmeans(model):
    """Generate explanation for K-Means."""
    return {
        'model_type': 'K-Means Clustering',
        'algorithm': 'K-Means',
        'description': 'Partitions data into k clusters by minimizing within-cluster variance.',
        'parameters': {
            'n_clusters': model.n_clusters,
            'init': model.init,
            'max_iter': model.max_iter,
            'n_init': model.n_init,
            'algorithm': model.algorithm
        },
        'cluster_info': {
            'n_clusters': model.n_clusters,
            'n_features': model.cluster_centers_.shape[1],
            'inertia': float(model.inertia_),
            'n_iter': int(model.n_iter_)
        },
        'cluster_centers': model.cluster_centers_.tolist(),
        'how_it_works': [
            '1. Randomly initializes k cluster centers',
            '2. Assigns each data point to nearest cluster center',
            '3. Recalculates cluster centers as mean of assigned points',
            '4. Repeats steps 2-3 until convergence or max iterations'
        ],
        'interpretation': 'Lower inertia indicates tighter, more compact clusters.'
    }


# ==================== HIERARCHICAL CLUSTERING ====================

def visualize_hierarchical(model):
    """Generate Hierarchical Clustering visualization."""
    n_clusters = model.n_clusters if model.n_clusters else 'Not specified'
    
    # Show cluster distribution if labels are available
    if hasattr(model, 'labels_') and model.labels_ is not None:
        unique_labels, counts = np.unique(model.labels_, return_counts=True)
        
        fig = go.Figure(data=[go.Bar(
            x=[f'Cluster {i}' for i in unique_labels],
            y=counts,
            marker=dict(color=counts, colorscale='Viridis', showscale=True)
        )])
        fig.update_layout(
            title=f'Hierarchical Clustering - Cluster Sizes ({len(unique_labels)} clusters)',
            xaxis_title='Cluster',
            yaxis_title='Number of Points',
            template='plotly_dark'
        )
    else:
        # Show configuration info
        fig = go.Figure()
        fig.add_annotation(
            text=f'<b>Hierarchical Clustering</b><br>Linkage: {model.linkage}<br>Clusters: {n_clusters}',
            showarrow=False,
            xref='paper', yref='paper',
            x=0.5, y=0.5,
            font=dict(size=16, color='white')
        )
        fig.update_layout(template='plotly_dark', title='Hierarchical Clustering Configuration')

    return {'type': 'plotly', 'data': fig.to_dict()}


def explain_hierarchical(model):
    """Generate explanation for Hierarchical Clustering."""
    explanation = {
        'model_type': 'Hierarchical Clustering (Agglomerative)',
        'algorithm': 'Agglomerative Hierarchical Clustering',
        'description': 'Builds a hierarchy of clusters using a bottom-up approach.',
        'parameters': {
            'n_clusters': model.n_clusters,
            'linkage': model.linkage,
            'affinity': model.affinity if hasattr(model, 'affinity') else model.metric
        },
        'how_it_works': [
            '1. Starts with each data point as its own cluster',
            '2. Iteratively merges the two closest clusters',
            '3. Uses linkage criterion to determine cluster distance',
            '4. Continues until desired number of clusters is reached'
        ],
        'linkage_types': {
            'ward': 'Minimizes variance within clusters',
            'complete': 'Maximum distance between cluster points',
            'average': 'Average distance between all pairs',
            'single': 'Minimum distance between cluster points'
        }
    }

    if hasattr(model, 'labels_') and model.labels_ is not None:
        unique_labels, counts = np.unique(model.labels_, return_counts=True)
        explanation['cluster_distribution'] = {
            'n_clusters_formed': len(unique_labels),
            'cluster_sizes': counts.tolist()
        }

    return explanation


# ==================== NAIVE BAYES ====================

def visualize_naive_bayes(model):
    """Generate Naive Bayes visualization."""
    if hasattr(model, 'classes_'):
        classes = model.classes_
        
        # Show class prior probabilities
        if hasattr(model, 'class_prior_'):
            priors = model.class_prior_
        elif hasattr(model, 'class_log_prior_'):
            priors = np.exp(model.class_log_prior_)
        else:
            priors = np.ones(len(classes)) / len(classes)

        fig = go.Figure(data=[go.Bar(
            x=[f'Class {c}' for c in classes],
            y=priors,
            text=[f'{p:.4f}' for p in priors],
            textposition='auto',
            marker=dict(color='lightblue')
        )])
        fig.update_layout(
            title='Naive Bayes - Class Prior Probabilities',
            xaxis_title='Class',
            yaxis_title='Prior Probability',
            template='plotly_dark'
        )
    else:
        fig = go.Figure()
        fig.add_annotation(
            text='<b>Naive Bayes Model</b><br>Class information not available',
            showarrow=False,
            xref='paper', yref='paper',
            x=0.5, y=0.5,
            font=dict(size=16, color='white')
        )
        fig.update_layout(template='plotly_dark')

    return {'type': 'plotly', 'data': fig.to_dict()}


def explain_naive_bayes(model):
    """Generate explanation for Naive Bayes."""
    model_name = type(model).__name__
    
    explanation = {
        'model_type': f'Naive Bayes ({model_name})',
        'algorithm': 'Naive Bayes',
        'description': 'Probabilistic classifier based on Bayes\' theorem with strong independence assumptions.',
        'variant': {
            'GaussianNB': 'Assumes features follow Gaussian (normal) distribution',
            'MultinomialNB': 'For discrete counts (e.g., word frequencies)',
            'BernoulliNB': 'For binary/boolean features'
        }.get(model_name, 'Unknown variant'),
        'how_it_works': [
            '1. Calculates prior probability of each class from training data',
            '2. Calculates likelihood of features given each class',
            '3. Assumes features are conditionally independent (naive assumption)',
            '4. Uses Bayes\' theorem to compute posterior probabilities',
            '5. Predicts class with highest posterior probability'
        ],
        'equation': 'P(class|features) ∝ P(class) × P(features|class)'
    }

    if hasattr(model, 'classes_'):
        explanation['classes'] = model.classes_.tolist()
        explanation['n_classes'] = len(model.classes_)

    if hasattr(model, 'class_prior_'):
        explanation['class_priors'] = model.class_prior_.tolist()
    elif hasattr(model, 'class_log_prior_'):
        explanation['class_log_priors'] = model.class_log_prior_.tolist()

    return explanation
