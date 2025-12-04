"""
Sample Model Generator for StellarBasin
This script creates sample models for testing the visualizer.
"""

import joblib
import numpy as np
from sklearn.datasets import load_iris, make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering
import os

# Create models directory
os.makedirs('../models', exist_ok=True)

print("🚀 Generating sample models for StellarBasin...\n")

# ==================== 1. DECISION TREE ====================
print("1️⃣  Creating Decision Tree Classifier...")
iris = load_iris()
X, y = iris.data, iris.target
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X, y)
joblib.dump(dt_model, '../models/decision_tree_classifier.pkl')
print("   ✅ Saved: decision_tree_classifier.pkl\n")

# ==================== 2. LINEAR REGRESSION (1D) ====================
print("2️⃣  Creating Linear Regression (1D)...")
X_1d = np.random.rand(100, 1) * 10
y_1d = 2.5 * X_1d.flatten() + 3 + np.random.randn(100) * 0.5
lr_1d = LinearRegression()
lr_1d.fit(X_1d, y_1d)
joblib.dump(lr_1d, '../models/linear_regression_1d.pkl')
print("   ✅ Saved: linear_regression_1d.pkl\n")

# ==================== 3. LINEAR REGRESSION (2D) ====================
print("3️⃣  Creating Linear Regression (2D)...")
X_2d = np.random.rand(100, 2) * 10
y_2d = 2 * X_2d[:, 0] + 3 * X_2d[:, 1] + 5 + np.random.randn(100) * 0.5
lr_2d = LinearRegression()
lr_2d.fit(X_2d, y_2d)
joblib.dump(lr_2d, '../models/linear_regression_2d.pkl')
print("   ✅ Saved: linear_regression_2d.pkl\n")

# ==================== 4. LOGISTIC REGRESSION ====================
print("4️⃣  Creating Logistic Regression...")
X_log, y_log = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                     n_informative=2, n_clusters_per_class=1, 
                                     random_state=42)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_log, y_log)
joblib.dump(log_reg, '../models/logistic_regression.pkl')
print("   ✅ Saved: logistic_regression.pkl\n")

# ==================== 5. SVM ====================
print("5️⃣  Creating SVM Classifier...")
X_svm, y_svm = make_classification(n_samples=150, n_features=2, n_redundant=0,
                                     n_informative=2, random_state=42)
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_svm, y_svm)
joblib.dump(svm_model, '../models/svm_classifier.pkl')
print("   ✅ Saved: svm_classifier.pkl\n")

# ==================== 6. KNN CLASSIFIER ====================
print("6️⃣  Creating KNN Classifier...")
knn_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_clf.fit(X, y)  # Using iris dataset
joblib.dump(knn_clf, '../models/knn_classifier.pkl')
print("   ✅ Saved: knn_classifier.pkl\n")

# ==================== 7. K-MEANS CLUSTERING ====================
print("7️⃣  Creating K-Means Clustering...")
X_kmeans, _ = make_blobs(n_samples=300, n_features=2, centers=4, 
                         cluster_std=0.6, random_state=42)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_kmeans)
joblib.dump(kmeans, '../models/kmeans_clustering.pkl')
print("   ✅ Saved: kmeans_clustering.pkl\n")

# ==================== 8. HIERARCHICAL CLUSTERING ====================
print("8️⃣  Creating Hierarchical Clustering...")
X_hier, _ = make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)
hier_model = AgglomerativeClustering(n_clusters=3, linkage='ward')
hier_model.fit(X_hier)
joblib.dump(hier_model, '../models/hierarchical_clustering.pkl')
print("   ✅ Saved: hierarchical_clustering.pkl\n")

# ==================== 9. NAIVE BAYES ====================
print("9️⃣  Creating Naive Bayes Classifier...")
nb_model = GaussianNB()
nb_model.fit(X, y)  # Using iris dataset
joblib.dump(nb_model, '../models/naive_bayes.pkl')
print("   ✅ Saved: naive_bayes.pkl\n")

# ==================== 10. DECISION TREE REGRESSOR ====================
print("🔟 Creating Decision Tree Regressor...")
X_reg, y_reg = make_regression(n_samples=100, n_features=4, noise=10, random_state=42)
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_reg.fit(X_reg, y_reg)
joblib.dump(dt_reg, '../models/decision_tree_regressor.pkl')
print("   ✅ Saved: decision_tree_regressor.pkl\n")

print("=" * 60)
print("✨ All sample models generated successfully!")
print("=" * 60)
print("\n📁 Models saved in: ../models/")
print("\n🎯 You can now upload these models to StellarBasin for testing!")
print("\nAvailable models:")
print("  1. decision_tree_classifier.pkl")
print("  2. linear_regression_1d.pkl")
print("  3. linear_regression_2d.pkl")
print("  4. logistic_regression.pkl")
print("  5. svm_classifier.pkl")
print("  6. knn_classifier.pkl")
print("  7. kmeans_clustering.pkl")
print("  8. hierarchical_clustering.pkl")
print("  9. naive_bayes.pkl")
print("  10. decision_tree_regressor.pkl")
