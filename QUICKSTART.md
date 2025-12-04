# 🚀 STELLARBASIN - Quick Start Guide

## ⚡ 3-Minute Setup

### Step 1: Start the Backend (Flask)
```bash
cd backend
python app.py
```
✅ Backend running on **http://localhost:5000**

### Step 2: Start the Frontend (React)
Open a **new terminal**:
```bash
cd frontend
npm start
```
✅ Frontend opens automatically on **http://localhost:3000**

### Step 3: Generate Sample Models (Optional)
```bash
cd trainer
python generate_sample_models.py
```
✅ Creates 10 sample models in `models/` folder

---

## 🎯 Upload Your First Model

### Option A: Use Sample Models
1. Run the sample generator (Step 3 above)
2. In the web app, click **"Choose File"**
3. Select any `.pkl` file from `models/` folder
4. Click **"Upload & Analyze"**

### Option B: Use Your Own Model
```python
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Train a model
X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Save it
joblib.dump(model, 'my_model.pkl')
```

Then upload `my_model.pkl` in the web app!

---

## 📊 What You'll See

### 1. **Model Type Detection**
Automatically identifies your model type

### 2. **Interactive Visualization**
- 3D plots for 2D regression
- Decision boundaries for classifiers
- Cluster centers for clustering
- Tree structures for decision trees

### 3. **Detailed Explanation**
- What the model does
- How it works (step-by-step)
- All parameters and settings
- Mathematical equations
- Interpretation tips

---

## 🎨 Supported Models

| Model | Icon | Visualization |
|-------|------|---------------|
| Decision Tree | 🌳 | Tree structure |
| Linear Regression | 📈 | Line/Surface plot |
| Logistic Regression | 📊 | Sigmoid/Contour |
| SVM | 🎯 | Support vectors |
| KNN | 🔍 | Class distribution |
| K-Means | 🔵 | Cluster centers |
| Hierarchical | 🌲 | Cluster sizes |
| Naive Bayes | 🎲 | Prior probabilities |

---

## 🐛 Troubleshooting

### Backend won't start
```bash
# Install dependencies
pip install flask flask-cors scikit-learn plotly matplotlib numpy pandas
```

### Frontend won't start
```bash
# Clear and reinstall
rm -rf node_modules
npm install
```

### Port already in use
Change the port in `backend/app.py`:
```python
app.run(debug=True, port=5001)  # Use 5001 instead
```

---

## 💡 Pro Tips

1. **Use sample models** to explore features first
2. **Try different model types** to see various visualizations
3. **Check the explanation panel** for learning insights
4. **Hover over charts** for interactive details
5. **Upload multiple models** to compare (one at a time)

---

## 📚 Learn More

- Full documentation: `README.md`
- Rebuild details: `REBUILD_SUMMARY.md`
- Sample code: `trainer/generate_sample_models.py`

---

## ✨ That's It!

You're ready to visualize and understand ML models like never before!

**Happy Exploring! 🎉**
