# Student Academic Performance Analysis
# Comprehensive Statistical Analysis for Scientific Article

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind, pearsonr, normaltest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# File upload functionality for Google Colab
from google.colab import files
import io

print("=== Student Academic Performance Analysis ===")
print("\nPlease upload your CSV dataset file:")
uploaded = files.upload()

# Load the dataset
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]))

print(f"\nDataset loaded successfully!")
print(f"Original dataset shape: {df.shape}")

# Display basic information about the dataset
print("\n=== DATASET OVERVIEW ===")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Data preprocessing and cleaning
print("\n=== DATA PREPROCESSING ===")

# Remove any duplicate rows
df = df.drop_duplicates()
print(f"Dataset shape after removing duplicates: {df.shape}")

# Handle missing values if any
df = df.dropna()

# Create binary target variable for final result
df['success'] = (df['final_result'] == 'Pass').astype(int)

# Select key columns for analysis (based on the provided image)
key_columns = [
    'gender', 'age', 'grade_level', 'math_score', 'reading_score', 
    'writing_score', 'attendance', 'parent_education', 'study_hours', 
    'internet_access', 'lunch_type', 'extracurricular', 'final_result', 'success'
]

# Keep only existing columns
available_columns = [col for col in key_columns if col in df.columns]
df_analysis = df[available_columns].copy()

print(f"Selected {len(available_columns)} columns for analysis")
print(f"Final dataset shape: {df_analysis.shape}")

# Create numerical encodings for categorical variables
le_dict = {}
categorical_cols = ['gender', 'parent_education', 'internet_access', 'lunch_type', 'extracurricular']

for col in categorical_cols:
    if col in df_analysis.columns:
        le_dict[col] = LabelEncoder()
        df_analysis[f'{col}_encoded'] = le_dict[col].fit_transform(df_analysis[col])

print("\n=== EXPLORATORY DATA ANALYSIS ===")

# 1. Distribution of Final Results
plt.figure(figsize=(15, 12))

plt.subplot(2, 3, 1)
result_counts = df_analysis['final_result'].value_counts()
plt.pie(result_counts.values, labels=result_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Final Results', fontsize=12, fontweight='bold')

# 2. Academic Scores Distribution
plt.subplot(2, 3, 2)
scores = ['math_score', 'reading_score', 'writing_score']
score_data = [df_analysis[score].dropna() for score in scores if score in df_analysis.columns]
score_labels = [score.replace('_', ' ').title() for score in scores if score in df_analysis.columns]

plt.boxplot(score_data, labels=score_labels)
plt.title('Distribution of Academic Scores', fontsize=12, fontweight='bold')
plt.ylabel('Score')
plt.xticks(rotation=45)

# 3. Gender vs Success Rate
if 'gender' in df_analysis.columns:
    plt.subplot(2, 3, 3)
    gender_success = df_analysis.groupby('gender')['success'].mean()
    gender_success.plot(kind='bar', color=['lightblue', 'lightcoral'])
    plt.title('Success Rate by Gender', fontsize=12, fontweight='bold')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)

# 4. Study Hours vs Performance
if 'study_hours' in df_analysis.columns:
    plt.subplot(2, 3, 4)
    plt.scatter(df_analysis['study_hours'], df_analysis['math_score'], alpha=0.6, color='green')
    plt.xlabel('Study Hours')
    plt.ylabel('Math Score')
    plt.title('Study Hours vs Math Performance', fontsize=12, fontweight='bold')

# 5. Attendance vs Success
if 'attendance' in df_analysis.columns:
    plt.subplot(2, 3, 5)
    plt.scatter(df_analysis['attendance'], df_analysis['success'], alpha=0.6, color='orange')
    plt.xlabel('Attendance (%)')
    plt.ylabel('Success (0=Fail, 1=Pass)')
    plt.title('Attendance vs Academic Success', fontsize=12, fontweight='bold')

# 6. Parent Education Impact
if 'parent_education' in df_analysis.columns:
    plt.subplot(2, 3, 6)
    parent_ed_success = df_analysis.groupby('parent_education')['success'].mean()
    parent_ed_success.plot(kind='bar', color='purple', alpha=0.7)
    plt.title('Success Rate by Parent Education', fontsize=12, fontweight='bold')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Additional detailed visualizations
plt.figure(figsize=(15, 10))

# Correlation heatmap
plt.subplot(2, 2, 1)
numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
correlation_matrix = df_analysis[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Numeric Variables', fontsize=12, fontweight='bold')

# Score distributions by success
plt.subplot(2, 2, 2)
if 'math_score' in df_analysis.columns:
    df_analysis.boxplot(column='math_score', by='final_result', ax=plt.gca())
    plt.title('Math Score Distribution by Final Result', fontsize=12, fontweight='bold')
    plt.suptitle('')

# Age distribution
plt.subplot(2, 2, 3)
if 'age' in df_analysis.columns:
    df_analysis['age'].hist(bins=10, alpha=0.7, color='skyblue')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Distribution', fontsize=12, fontweight='bold')

# Grade level analysis
plt.subplot(2, 2, 4)
if 'grade_level' in df_analysis.columns:
    grade_success = df_analysis.groupby('grade_level')['success'].mean()
    grade_success.plot(kind='bar', color='lightgreen')
    plt.title('Success Rate by Grade Level', fontsize=12, fontweight='bold')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

print("\n=== STATISTICAL HYPOTHESIS TESTING ===")

# Hypothesis 1: Gender differences in academic performance
print("\n1. HYPOTHESIS TEST: Gender Differences in Math Performance")
if 'gender' in df_analysis.columns and 'math_score' in df_analysis.columns:
    male_scores = df_analysis[df_analysis['gender'] == 'Male']['math_score'].dropna()
    female_scores = df_analysis[df_analysis['gender'] == 'Female']['math_score'].dropna()
    
    # Two-sample t-test
    t_stat, p_value = ttest_ind(male_scores, female_scores)
    
    print(f"Male Math Score - Mean: {male_scores.mean():.2f}, Std: {male_scores.std():.2f}")
    print(f"Female Math Score - Mean: {female_scores.mean():.2f}, Std: {female_scores.std():.2f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significance level (α = 0.05): {'Significant' if p_value < 0.05 else 'Not Significant'}")

# Hypothesis 2: Parent education impact on success
print("\n2. HYPOTHESIS TEST: Parent Education Impact on Success")
if 'parent_education' in df_analysis.columns:
    contingency_table = pd.crosstab(df_analysis['parent_education'], df_analysis['final_result'])
    print("\nContingency Table:")
    print(contingency_table)
    
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_val:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"Significance level (α = 0.05): {'Significant' if p_val < 0.05 else 'Not Significant'}")

# Hypothesis 3: Study hours effect on performance
print("\n3. HYPOTHESIS TEST: Study Hours Correlation with Performance")
if 'study_hours' in df_analysis.columns and 'math_score' in df_analysis.columns:
    correlation, p_val = pearsonr(df_analysis['study_hours'].dropna(), 
                                 df_analysis['math_score'].dropna())
    print(f"Pearson correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_val:.4f}")
    print(f"Significance level (α = 0.05): {'Significant' if p_val < 0.05 else 'Not Significant'}")

# Hypothesis 4: ANOVA test for grade level differences
print("\n4. HYPOTHESIS TEST: Grade Level Differences in Performance (ANOVA)")
if 'grade_level' in df_analysis.columns and 'math_score' in df_analysis.columns:
    grade_groups = [group['math_score'].dropna() for name, group in df_analysis.groupby('grade_level')]
    f_stat, p_val = f_oneway(*grade_groups)
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    print(f"Significance level (α = 0.05): {'Significant' if p_val < 0.05 else 'Not Significant'}")

# Normality tests
print("\n5. NORMALITY TESTS")
numeric_columns = ['math_score', 'reading_score', 'writing_score', 'attendance']
for col in numeric_columns:
    if col in df_analysis.columns:
        stat, p_val = normaltest(df_analysis[col].dropna())
        print(f"{col}: D'Agostino normality test p-value = {p_val:.4f}")
        print(f"  Distribution is {'Normal' if p_val > 0.05 else 'Not Normal'} (α = 0.05)")

print("\n=== MACHINE LEARNING ANALYSIS ===")

# Prepare data for machine learning
# Select features for prediction
feature_columns = []
for col in df_analysis.columns:
    if col.endswith('_encoded') or col in ['age', 'grade_level', 'math_score', 'reading_score', 
                                          'writing_score', 'attendance', 'study_hours']:
        if col in df_analysis.columns:
            feature_columns.append(col)

# Remove rows with missing values in selected features
ml_data = df_analysis[feature_columns + ['success']].dropna()

if len(ml_data) > 50:  # Ensure we have enough data
    X = ml_data[feature_columns]
    y = ml_data['success']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Logistic Regression
    print("\n1. LOGISTIC REGRESSION")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, lr_pred))
    
    # Random Forest
    print("\n2. RANDOM FOREST")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Random Forest):")
    print(feature_importance)
    
    # Visualization of results
    plt.figure(figsize=(12, 8))
    
    # Model comparison
    plt.subplot(2, 2, 1)
    models = ['Logistic Regression', 'Random Forest']
    accuracies = [lr_accuracy, rf_accuracy]
    plt.bar(models, accuracies, color=['blue', 'green'], alpha=0.7)
    plt.title('Model Performance Comparison', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Feature importance plot
    plt.subplot(2, 2, 2)
    top_features = feature_importance.head(min(10, len(feature_importance)))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title('Top Feature Importance', fontsize=12, fontweight='bold')
    plt.xlabel('Importance')
    
    # Confusion matrix for Random Forest
    plt.subplot(2, 2, 3)
    cm = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Random Forest)', fontsize=12, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Prediction probabilities distribution
    plt.subplot(2, 2, 4)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    plt.hist(rf_proba, bins=20, alpha=0.7, color='orange')
    plt.title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Probability of Success')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

print("\n=== SUMMARY STATISTICS TABLE ===")

# Create comprehensive summary table
summary_stats = pd.DataFrame()

numeric_cols = ['math_score', 'reading_score', 'writing_score', 'attendance', 'study_hours', 'age']
for col in numeric_cols:
    if col in df_analysis.columns:
        summary_stats[col] = [
            df_analysis[col].count(),
            df_analysis[col].mean(),
            df_analysis[col].std(),
            df_analysis[col].min(),
            df_analysis[col].quantile(0.25),
            df_analysis[col].median(),
            df_analysis[col].quantile(0.75),
            df_analysis[col].max()
        ]

summary_stats.index = ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
print("\nDescriptive Statistics Summary:")
print(summary_stats.round(2))

# Success rate by categorical variables
print("\n=== SUCCESS RATES BY CATEGORIES ===")
categorical_vars = ['gender', 'parent_education', 'internet_access', 'lunch_type', 'extracurricular']

for var in categorical_vars:
    if var in df_analysis.columns:
        success_rates = df_analysis.groupby(var)['success'].agg(['count', 'mean', 'std']).round(3)
        print(f"\nSuccess Rate by {var.replace('_', ' ').title()}:")
        print(success_rates)

print("\n=== ANALYSIS COMPLETE ===")
print("\nKey Findings Summary:")
print("1. Exploratory data analysis completed with multiple visualizations")
print("2. Statistical hypothesis tests performed for key relationships")
print("3. Machine learning models trained and evaluated")
print("4. Comprehensive statistical summaries generated")
print("\nAll visualizations and statistical tests are ready for inclusion in your scientific article.")
print("Remember to interpret the results in the context of your research questions!")

# Export key results to CSV for further analysis
results_summary = {
    'Total_Students': len(df_analysis),
    'Success_Rate': df_analysis['success'].mean(),
    'Average_Math_Score': df_analysis['math_score'].mean() if 'math_score' in df_analysis.columns else 'N/A',
    'Average_Attendance': df_analysis['attendance'].mean() if 'attendance' in df_analysis.columns else 'N/A',
}

print(f"\nKey Metrics:")
for key, value in results_summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.3f}")
    else:
        print(f"{key}: {value}")