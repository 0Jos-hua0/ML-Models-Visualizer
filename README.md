# 🌟 STELLARBASIN - ML Model Visualizer & Explainer

**StellarBasin** is an advanced web application that visualizes and explains machine learning models with beautiful, interactive charts and comprehensive explanations.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![React](https://img.shields.io/badge/react-19.1.0-blue)
![Flask](https://img.shields.io/badge/flask-latest-red)

---

## ✨ Features

### 🎯 Supported Models

StellarBasin supports **7 major machine learning model types**:

1. **🌳 Decision Tree** (Classifier & Regressor)
   - Interactive tree structure visualization
   - Node and leaf information
   - Feature importance analysis

2. **📈 Linear Regression** (1D & 2D)
   - 1D: Line plot with slope and intercept
   - 2D: 3D surface visualization
   - Multi-dimensional: Coefficient bar charts

3. **📊 Logistic Regression**
   - 1D: Sigmoid curve visualization
   - 2D: Decision boundary contour plots
   - Probability distributions

4. **🎯 Support Vector Machine (SVM)**
   - Support vector visualization
   - Kernel information
   - Decision boundary analysis

5. **🔍 K-Nearest Neighbors (KNN)**
   - Class distribution charts
   - Neighbor configuration
   - Distance metrics

6. **🔵 K-Means Clustering**
   - Cluster center visualization (2D/3D)
   - Cluster heatmaps
   - Inertia and convergence info

7. **🌲 Hierarchical Clustering**
   - Cluster size distribution
   - Linkage method explanation
   - Dendrogram information

8. **🎲 Naive Bayes** (Gaussian, Multinomial, Bernoulli)
   - Prior probability visualization
   - Variant-specific explanations
   - Bayes theorem breakdown

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Node.js 14+**
- **npm or yarn**

### Installation

#### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd StellarBasin
```

#### 2. Backend Setup (Flask)

```bash
# Navigate to backend
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Frontend Setup (React)

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install
```

---

## 🎮 Running the Application

### Start Backend Server

```bash
cd backend
python app.py
```

The Flask server will start on **http://localhost:5000**

### Start Frontend Server

Open a **new terminal** and run:

```bash
cd frontend
npm start
```

The React app will open automatically on **http://localhost:3000**

---

## 📖 How to Use

### Step 1: Prepare Your Model

Train and save your scikit-learn model using `joblib`:

```python
import joblib
from sklearn.tree import DecisionTreeClassifier

# Train your model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'my_model.pkl')
```

### Step 2: Upload Model

1. Open the StellarBasin web app
2. Click **"Choose File"** and select your `.pkl` model file
3. Click **"Upload & Analyze"**

### Step 3: Explore Results

The app will display:

- **📊 Interactive Visualizations** - Plotly charts or matplotlib images
- **🧠 Detailed Explanations** - How the model works, parameters, equations
- **📐 Mathematical Formulas** - The underlying equations
- **⚙️ Model Parameters** - All hyperparameters and settings
- **💡 Interpretations** - What the results mean

---

## 🎨 Visualization Examples

### Decision Tree
- Full tree structure with nodes and leaves
- Color-coded by class or value
- Feature importance bars

### Linear Regression (2D)
- 3D surface plot showing prediction plane
- Interactive rotation and zoom
- Coefficient annotations

### K-Means Clustering
- 2D/3D scatter plots of cluster centers
- Heatmaps for high-dimensional data
- Cluster statistics

### SVM
- Support vector highlights
- Decision boundary visualization
- Margin representation

---

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **scikit-learn** - ML model support
- **Plotly** - Interactive visualizations
- **Matplotlib** - Static visualizations
- **NumPy** - Numerical computations

### Frontend
- **React 19** - UI framework
- **Plotly.js** - Interactive charts
- **Axios** - HTTP requests
- **CSS3** - Styling and animations

---

## 📁 Project Structure

```
StellarBasin/
├── backend/
│   ├── app.py                 # Flask server
│   ├── stellarbasin.py        # Model visualization logic
│   ├── requirements.txt       # Python dependencies
│   └── uploads/               # Temporary upload folder
├── frontend/
│   ├── src/
│   │   ├── App.js            # Main application
│   │   ├── components/
│   │   │   ├── ModelUploader.js      # Upload component
│   │   │   └── ExplanationPanel.js   # Explanation display
│   │   └── assets/           # Images and resources
│   ├── package.json          # Node dependencies
│   └── public/               # Static files
├── models/                   # Sample models (optional)
└── README.md                # This file
```

---

## 🧪 Testing with Sample Models

You can test the app with sample models from the `models/` directory or create your own:

```python
# Example: Create a simple Decision Tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Save model
joblib.dump(model, 'iris_decision_tree.pkl')
```

---

## 🎯 Supported File Formats

- `.pkl` (Pickle)
- `.joblib` (Joblib)
- `.pickle` (Pickle variant)

---

## 🐛 Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError`
```bash
# Solution: Install missing dependencies
pip install -r backend/requirements.txt
```

**Problem:** Port 5000 already in use
```bash
# Solution: Change port in app.py
app.run(debug=True, port=5001)
```

### Frontend Issues

**Problem:** `npm start` fails
```bash
# Solution: Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Problem:** CORS errors
```bash
# Solution: Ensure Flask-CORS is installed
pip install flask-cors
```

---

## 🔮 Future Enhancements

- [ ] Support for ensemble models (Random Forest, Gradient Boosting)
- [ ] Neural network visualization
- [ ] Model comparison feature
- [ ] Export visualizations as images
- [ ] Batch model upload
- [ ] Model performance metrics
- [ ] Custom color themes

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for the Machine Learning community**

🌟 Star this repo if you find it helpful!
