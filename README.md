# Student Academic Performance Analysis

This project analyzes a dataset of student academic performance using statistical methods and machine learning models.

---

## 📁 Project Structure

```
code/
  ├── analysis.py          # Full analysis script
  └── run_analysis.py      # Script to run full analysis from terminal

data/
  └── student_info.csv     # Input dataset (1000 students)

report/
  └── Python Task.pdf      # Full academic report with results, graphs, and conclusions

analysis_full_notebook.ipynb  # Interactive Jupyter Notebook for full analysis
requirements.txt              # Python dependencies
```

---

## 🚀 How to Run

### Option 1: Run the Jupyter Notebook  
Open `analysis_full_notebook.ipynb` in Jupyter (locally or on GitHub) to explore the full analysis.

### Option 2: Run as Python Script  
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run:
```bash
python code/run_analysis.py
```

---

## ⚙️ Features

- Exploratory Data Analysis (EDA)
- Statistical Hypothesis Testing
- Machine Learning Classification
- Visualization with Seaborn & Matplotlib

---

## 📊 Main Findings

- Gender significantly affects math performance (p = 0.0318)
- Parental education has no significant effect on success (p = 0.6830)
- Study hours are not significantly correlated with performance
- Random Forest achieved 56.5% accuracy predicting student success

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🔗 Citation

If you use this code or report, please cite:

**"Student Academic Performance Analysis", Yoni Fluk, 2025**