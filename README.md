# Student Academic Performance Analysis

This project analyzes a dataset of student academic performance using statistical methods and machine learning.

## 📁 Project Structure

- `code/analysis.py`: The full analysis code (originally from `Code.txt`)
- `data/student_info.csv`: The input dataset (1000 students)
- `report/Python_Task.pdf`: Full academic report with results, graphs, and conclusions

## 🚀 How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `code/analysis.py` and `data/student_info.csv`
3. Run all cells. Visualizations and analysis will appear directly.
4. The code performs:
   - Exploratory data analysis
   - Statistical hypothesis testing
   - Machine learning classification
   - Summary statistics

## 🛠 Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scipy
- scikit-learn

Install with:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## 📊 Main Findings

- Gender significantly affects math performance (p = 0.0318)
- Parental education has **no significant effect** on success (p = 0.6830)
- Study hours are **not significantly correlated** with performance
- Random Forest achieved 56.5% accuracy predicting student success

## 🔗 Citation

If you use this code or report, please cite:

> "Student Academic Performance Analysis", Yoni Fluk, 2025

## 📄 License

This project is released under the MIT License.