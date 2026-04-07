# 🤖 AI Data Science Assistant

An advanced AI-powered data analysis platform with comprehensive statistical analysis, interactive visualizations, and machine learning capabilities.

## 🌟 Features

### 📊 Comprehensive Data Analysis (15 Sections)
- **Dataset Overview** - Complete data profiling with memory usage
- **Data Quality Metrics** - 5-dimension quality scoring (A-D grading)
- **Statistical Analysis** - Descriptive stats, skewness, kurtosis, correlations
- **ML Readiness Assessment** - PCA analysis and readiness scoring
- **Advanced Statistical Tests** - Normality tests (Shapiro-Wilk, Anderson-Darling)
- **Feature Importance** - Random Forest + Mutual Information analysis
- **Outlier Detection** - Multi-method detection (IQR, Z-Score, MAD)
- **Time Series Analysis** - Automatic detection with rolling statistics
- **Categorical Analysis** - Chi-Square independence tests
- **Distribution Comparison** - Violin plots and KS tests
- **Correlation Networks** - Threshold-based correlation discovery
- **AI Insights** - Context-aware recommendations
- **Interactive Explorer** - Custom scatter plots with trendlines
- **Quality Report** - Comprehensive checklist with grades
- **Export Capabilities** - Downloadable CSV reports

### 🎨 Interactive Visualizations
- **Plotly Integration** - All charts are interactive (zoom, pan, hover)
- **Heatmaps** - Correlation and contingency tables
- **Time Series Plots** - Trend analysis with moving averages
- **Violin Plots** - Distribution comparisons
- **Box Plots** - Outlier detection
- **Scatter Plots** - Custom X/Y with trendlines
- **Q-Q Plots** - Normality assessment
- **Radar Charts** - ML readiness visualization

### 🤖 Machine Learning
- **Multiple Algorithms** - Linear Regression, Random Forest, Gradient Boosting, SVR, Ridge, Lasso
- **Classification Support** - Logistic Regression, SVM, Decision Trees
- **Cross-Validation** - Configurable K-fold CV
- **Model Comparison** - Side-by-side performance metrics
- **Feature Selection** - Automated importance ranking
- **Training Metrics** - MSE, MAE, R², Accuracy, training time

### 🔒 Security
- **Data Encryption** - Fernet encryption for uploaded files
- **Secure Processing** - Encrypted data handling throughout pipeline

### 🧠 AI-Powered Insights
- **Gemini AI Integration** - Advanced natural language analysis
- **Context-Aware** - Receives full dataset statistics
- **Custom Questions** - Ask anything about your data
- **Variable Analysis** - Deep dive into individual columns
- **Actionable Recommendations** - Specific next steps

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run fast_app.py
```

Or with XSRF protection disabled:
```bash
streamlit run fast_app.py --server.enableXsrfProtection false
```

### Usage

1. **Upload CSV** - Click "Let's get started" and upload your CSV file
2. **Choose Analysis** - Select from sidebar:
   - Full Data Analysis (15 comprehensive sections)
   - Variable Analysis (deep dive into specific columns)
   - Custom Question (ask AI about your data)
   - Model Training (build ML models)
3. **Explore Results** - Interact with visualizations and insights
4. **Export Reports** - Download analysis summaries

## 📋 Requirements

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
scipy
cryptography
google-generativeai
```

## 🎯 Use Cases

- **Data Quality Assessment** - Comprehensive quality scoring and reporting
- **Exploratory Data Analysis** - 15-section deep dive into your data
- **Statistical Testing** - Normality, independence, and distribution tests
- **Feature Engineering** - Importance analysis and correlation detection
- **Outlier Detection** - Multi-method anomaly identification
- **Time Series Analysis** - Trend detection and rolling statistics
- **Machine Learning** - Automated model training and comparison
- **Data Documentation** - Exportable analysis reports

## 📚 Documentation

- **WHATS_NEW.md** - Overview of all new features
- **FEATURE_GUIDE.md** - Detailed section-by-section guide
- **ENHANCED_FEATURES.md** - Technical details and specifications

## 🎓 Example Workflow

1. Upload your dataset
2. Review **Data Quality Report** (Section 14) for overall grade
3. Check **Statistical Analysis** (Section 3) for distributions
4. Run **Normality Tests** (Section 5) to validate assumptions
5. Identify **Feature Importance** (Section 6) for modeling
6. Detect **Outliers** (Section 7) and decide on treatment
7. Review **AI Insights** (Section 12) for recommendations
8. **Export Report** (Section 15) for documentation
9. Build models in **Model Training** section

## 🔬 Advanced Features

### Statistical Tests
- Shapiro-Wilk normality test
- Anderson-Darling test
- Kolmogorov-Smirnov test
- Chi-Square independence test
- Q-Q plots for visual assessment

### Quality Metrics
- Completeness score
- Uniqueness score
- Consistency score
- Data type validity
- Outlier assessment
- Overall A-D grading

### ML Readiness
- PCA dimensionality analysis
- Sample size adequacy
- Feature diversity score
- Correlation health
- Overall readiness (0-100%)

## 🎨 Visualization Types

- Interactive scatter plots
- Correlation heatmaps
- Time series with moving averages
- Violin plots for distributions
- Box plots for outliers
- Q-Q plots for normality
- Bar charts for importance
- Radar charts for readiness
- Contingency table heatmaps

## 💡 Tips

- **Large Datasets**: Tool automatically uses sampling for expensive operations
- **Missing Values**: Check Section 2 for analysis and Section 12 for recommendations
- **Outliers**: Section 7 provides 3 detection methods for robust identification
- **Time Series**: Automatically detected if date columns present
- **Feature Selection**: Use Section 6 and 11 for correlation and importance
- **Quality Issues**: Follow recommendations in Section 12 and end of analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Powered by Google Gemini AI
- Built with Streamlit
- Visualizations by Plotly
- ML by scikit-learn

## 📞 Support

For questions or issues, please open an issue on GitHub.

---

**Made with ❤️ by Ana**

