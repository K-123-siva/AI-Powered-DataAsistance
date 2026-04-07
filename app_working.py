import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from cryptography.fernet import Fernet
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import google.generativeai as genai
from scipy import stats
import time

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyCYebsFl86S5cHy5kjnjQXy_KN2LtryvK8"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
@st.cache_resource
def get_gemini_model():
    try:
        models = genai.list_models()
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name
                st.sidebar.success(f"Using model: {model_name}")
                return genai.GenerativeModel(model_name)
        
        fallback_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'models/gemini-pro']
        for model_name in fallback_models:
            try:
                return genai.GenerativeModel(model_name)
            except:
                continue
                
        return None
    except Exception as e:
        st.sidebar.error(f"Model initialization error: {str(e)}")
        return None

# Initialize encryption key and cipher
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def generate_text(prompt, data_context=""):
    """Generate intelligent responses using Gemini API"""
    try:
        model = get_gemini_model()
        
        if model is None:
            raise Exception("Could not initialize Gemini model")
        
        enhanced_prompt = f"""
        You are an expert data scientist assistant. Analyze the specific data provided and give unique insights.
        
        Analysis Request: {prompt}
        
        {f"Specific Data Context: {data_context}" if data_context else ""}
        
        Requirements:
        - Provide specific insights based on the actual data values provided
        - Use the exact variable name and statistics in your analysis
        - Give actionable recommendations
        - Format with bullet points for clarity
        - Keep response focused and under 150 words
        - Be specific to this particular variable and its characteristics
        """
        
        response = model.generate_content(enhanced_prompt)
        return response.text
    
    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
        st.info("Using fallback analysis...")
        
        if "dataset" in prompt.lower() or "dataframe" in prompt.lower():
            return f"""
            **Data Analysis Summary:**
            Based on your question: "{prompt}"
            
            I'm currently unable to connect to the AI service, but here's what I can tell you from the data context:
            {data_context if data_context else "Please check the visualizations and statistics above for insights."}
            
            **Recommendations:**
            • Review the data overview and statistics
            • Check for missing values and data quality issues  
            • Examine the visualizations for patterns
            • Consider the suggested machine learning models
            """
        
        return f"""
        **Analysis Note:** 
        I'm currently unable to connect to the AI service for: "{prompt}"
        
        Please refer to the visualizations and statistics displayed above for insights about your data.
        
        {data_context if data_context else ""}
        """

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    try:
        decrypted_data = cipher_suite.decrypt(encrypted_data).decode()
        return decrypted_data
    except Exception as e:
        st.error(f"Decryption failed: {e}")
        raise

# Title
st.title('🤖 AI Assistant for Data Science')

# Welcoming message
st.write("Hello, 👋 I am your AI Assistant, and I am here to help you with your data science projects.")

# Explanation sidebar
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with a CSV File.*')
    st.caption('''**Upload a CSV file to begin. I will analyze the data, provide insights, generate visualizations, 
    and suggest appropriate machine learning models to tackle your problem. Let's dive into your data science journey!**
    ''')
    
    st.divider()
    
    st.caption("<p style ='text-align:center'> made with ❤️ by Ana</p>", unsafe_allow_html=True)

# Initialize the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    
    if user_csv is not None:
        # Encrypt the file for demonstration
        file_content = user_csv.read()
        encrypted_file = encrypt_data(file_content.decode('utf-8'))
        
        # Notify user about encryption
        st.write("Your data has been successfully encrypted.")
        
        try:
            decrypted_file = decrypt_data(encrypted_file)
            df = pd.read_csv(io.StringIO(decrypted_file), low_memory=False)
        except Exception as e:
            st.error(f"An error occurred during decryption: {e}")
            st.stop()
        
        # Enhanced Full Data Analysis Function with Evaluation Metrics
        def function_agent():
            st.title("📊 Comprehensive Data Analysis Report")
            
            # Dataset Overview
            st.header("1. 📋 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            with col4:
                st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            
            # Data Quality Evaluation Metrics
            st.header("2. 🎯 Data Quality Evaluation Metrics")
            
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            completeness_score = ((total_cells - missing_cells) / total_cells) * 100
            
            duplicate_count = df.duplicated().sum()
            uniqueness_score = ((df.shape[0] - duplicate_count) / df.shape[0]) * 100
            
            overall_quality = (completeness_score + uniqueness_score) / 2
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Completeness", f"{completeness_score:.1f}%")
            with col2:
                st.metric("Uniqueness", f"{uniqueness_score:.1f}%")
            with col3:
                quality_color = "🟢" if overall_quality >= 85 else "🟡" if overall_quality >= 70 else "🔴"
                st.metric("Overall Quality", f"{quality_color} {overall_quality:.1f}%")
            
            # Statistical Analysis
            st.header("3. 📈 Statistical Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                st.subheader("🔢 Numeric Variables Summary")
                st.dataframe(df[numeric_cols].describe())
                
                # Advanced Statistical Metrics
                st.subheader("📊 Advanced Statistical Metrics")
                advanced_stats = pd.DataFrame({
                    'Skewness': df[numeric_cols].skew(),
                    'Kurtosis': df[numeric_cols].kurtosis(),
                    'Variance': df[numeric_cols].var(),
                    'Std Dev': df[numeric_cols].std(),
                    'Range': df[numeric_cols].max() - df[numeric_cols].min(),
                    'IQR': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25),
                    'CV (%)': (df[numeric_cols].std() / df[numeric_cols].mean() * 100).round(2)
                })
                st.dataframe(advanced_stats)
                
                # Outlier Detection
                st.subheader("🔍 Outlier Detection (IQR Method)")
                outlier_summary = []
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_pct = (outlier_count / len(df)) * 100
                    
                    outlier_summary.append({
                        'Column': col,
                        'Outlier Count': outlier_count,
                        'Outlier %': f"{outlier_pct:.2f}%",
                        'Lower Bound': f"{lower_bound:.2f}",
                        'Upper Bound': f"{upper_bound:.2f}"
                    })
                
                outlier_df = pd.DataFrame(outlier_summary)
                st.dataframe(outlier_df)
                
                # Correlation analysis
                if len(numeric_cols) > 1:
                    st.subheader("🔗 Correlation Analysis")
                    corr_matrix = df[numeric_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                               center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
                    ax.set_title('Correlation Matrix (Lower Triangle)', fontsize=14, fontweight='bold')
                    st.pyplot(fig)
                    
                    # High Correlation Pairs
                    st.write("**🚨 High Correlation Pairs (|r| > 0.7):**")
                    high_corr = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.7:
                                high_corr.append({
                                    'Variable 1': corr_matrix.columns[i],
                                    'Variable 2': corr_matrix.columns[j],
                                    'Correlation': f"{corr_matrix.iloc[i, j]:.3f}"
                                })
                    
                    if high_corr:
                        st.dataframe(pd.DataFrame(high_corr))
                    else:
                        st.success("✅ No high correlations detected (good for ML models!)")
            
            # Categorical Variables Analysis
            if categorical_cols:
                st.subheader("📝 Categorical Variables Analysis")
                
                cat_summary = []
                for col in categorical_cols:
                    unique_count = df[col].nunique()
                    most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'
                    most_common_freq = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                    most_common_pct = (most_common_freq / len(df)) * 100
                    
                    cat_summary.append({
                        'Column': col,
                        'Unique Values': unique_count,
                        'Most Common': most_common,
                        'Frequency': most_common_freq,
                        'Percentage': f"{most_common_pct:.1f}%"
                    })
                
                cat_df = pd.DataFrame(cat_summary)
                st.dataframe(cat_df)
                
                # Visualize top categorical columns
                for col in categorical_cols[:3]:
                    st.write(f"**Distribution of {col}:**")
                    value_counts = df[col].value_counts().head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    value_counts.plot(kind='barh', ax=ax, color='skyblue')
                    ax.set_title(f'Top 10 Values in {col}')
                    ax.set_xlabel('Frequency')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Visualizations
            st.header("4. 📊 Advanced Data Visualizations")
            
            # Missing values analysis
            if df.isnull().sum().sum() > 0:
                st.subheader("❌ Missing Values Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Missing values heatmap
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(df.isnull(), cbar=True, cmap='viridis', ax=ax)
                    ax.set_title('Missing Values Heatmap')
                    st.pyplot(fig)
                
                with col2:
                    # Missing values bar chart
                    missing_data = df.isnull().sum()
                    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    missing_data.plot(kind='bar', ax=ax, color='coral')
                    ax.set_title('Missing Values by Column')
                    ax.set_ylabel('Count')
                    ax.set_xlabel('Columns')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.success("✅ No missing values in the dataset!")
            
            # Distribution Analysis
            if numeric_cols:
                st.subheader("📈 Distribution Analysis")
                
                # Histograms with KDE
                n_cols = min(len(numeric_cols), 4)
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.ravel()
                
                for i, col in enumerate(numeric_cols[:4]):
                    df[col].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}', fontweight='bold')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Box plots for outlier visualization
                st.subheader("📦 Box Plots (Outlier Detection)")
                
                fig, axes = plt.subplots(1, min(len(numeric_cols), 4), figsize=(15, 5))
                if len(numeric_cols) == 1:
                    axes = [axes]
                
                for i, col in enumerate(numeric_cols[:4]):
                    df.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'{col}')
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Pairplot for relationships (if not too many columns)
                if 2 <= len(numeric_cols) <= 5:
                    st.subheader("🔗 Pairplot (Variable Relationships)")
                    pairplot_fig = sns.pairplot(df[numeric_cols], diag_kind='kde', plot_kws={'alpha': 0.6})
                    pairplot_fig.fig.suptitle('Pairplot of Numeric Variables', y=1.02)
                    st.pyplot(pairplot_fig.fig)
            
            # Data Quality Report
            st.header("5. 📋 Data Quality Report")
            
            quality_issues = []
            recommendations = []
            
            # Check for missing values
            if missing_cells > 0:
                quality_issues.append(f"⚠️ {missing_cells} missing values ({(missing_cells/total_cells)*100:.1f}% of data)")
                recommendations.append("🔧 Handle missing values using imputation or removal")
            
            # Check for duplicates
            if duplicate_count > 0:
                quality_issues.append(f"⚠️ {duplicate_count} duplicate rows ({(duplicate_count/len(df))*100:.1f}% of data)")
                recommendations.append("🗑️ Remove or investigate duplicate records")
            
            # Check for high correlations
            if len(numeric_cols) > 1:
                high_corr_count = sum(1 for i in range(len(corr_matrix.columns)) 
                                     for j in range(i+1, len(corr_matrix.columns)) 
                                     if abs(corr_matrix.iloc[i, j]) > 0.8)
                if high_corr_count > 0:
                    quality_issues.append(f"⚠️ {high_corr_count} pairs of highly correlated features (>0.8)")
                    recommendations.append("📊 Consider feature selection or dimensionality reduction")
            
            # Check for outliers
            total_outliers = sum(int(row['Outlier Count']) for row in outlier_summary if 'outlier_summary' in locals())
            if 'outlier_summary' in locals() and total_outliers > 0:
                quality_issues.append(f"⚠️ {total_outliers} outliers detected across numeric columns")
                recommendations.append("🔍 Investigate and handle outliers appropriately")
            
            # Display quality issues
            if quality_issues:
                st.write("**⚠️ Data Quality Issues:**")
                for issue in quality_issues:
                    st.write(issue)
            else:
                st.success("✅ No major data quality issues detected!")
            
            # Display recommendations
            if recommendations:
                st.write("**💡 Recommendations:**")
                for rec in recommendations:
                    st.write(rec)
            
            # ML Model Suggestions
            st.header("6. 🤖 ML Model Suggestions & Readiness")
            
            # Calculate ML readiness score
            ml_readiness_factors = {
                'Data Completeness': completeness_score,
                'Data Uniqueness': uniqueness_score,
                'Feature Count': min(100, (len(numeric_cols) / max(1, len(df.columns))) * 100),
                'Sample Size': min(100, (len(df) / 1000) * 100)
            }
            
            ml_readiness = sum(ml_readiness_factors.values()) / len(ml_readiness_factors)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                readiness_color = "🟢" if ml_readiness >= 80 else "🟡" if ml_readiness >= 60 else "🔴"
                st.metric("ML Readiness Score", f"{readiness_color} {ml_readiness:.1f}%")
            
            with col2:
                st.metric("Numeric Features", len(numeric_cols))
            
            with col3:
                st.metric("Categorical Features", len(categorical_cols))
            
            # Model recommendations
            if len(numeric_cols) >= 2:
                st.success("✅ Dataset is suitable for machine learning!")
                
                st.write("**📊 Recommended Models Based on Your Data:**")
                
                # Regression models
                if len(numeric_cols) >= 2:
                    st.write("**For Regression Tasks:**")
                    st.write("• 🎯 **Linear Regression** - Good for linear relationships")
                    st.write("• 🌲 **Random Forest** - Handles non-linear patterns well")
                    st.write("• 🚀 **Gradient Boosting** - High accuracy, handles complex patterns")
                    st.write("• 📈 **Ridge/Lasso** - Good when features are correlated")
                
                # Classification models
                if categorical_cols:
                    st.write("**For Classification Tasks:**")
                    st.write("• 📊 **Logistic Regression** - Fast and interpretable")
                    st.write("• 🌲 **Random Forest Classifier** - Robust and accurate")
                    st.write("• 🚀 **Gradient Boosting Classifier** - High performance")
                    st.write("• 🎯 **Support Vector Machine** - Good for complex boundaries")
                
                # Advanced suggestions
                st.write("**🔬 Advanced Techniques:**")
                if len(numeric_cols) > 10:
                    st.write("• 📐 **PCA** - Reduce dimensionality")
                if len(df) > 10000:
                    st.write("• ⚡ **Feature Selection** - Improve training speed")
                if total_outliers > len(df) * 0.05:
                    st.write("• 🛡️ **Robust Scaling** - Handle outliers better")
            else:
                st.warning("⚠️ Need at least 2 numeric features for effective machine learning")
                st.info("💡 Consider feature engineering or encoding categorical variables")
            
            # AI-Powered Insights
            st.header("7. 🤖 AI-Powered Insights")
            
            dataset_summary = f"""
            Dataset: {df.shape[0]} rows, {df.shape[1]} columns
            Quality Score: {overall_quality:.1f}%
            ML Readiness: {ml_readiness:.1f}%
            Numeric Features: {len(numeric_cols)}
            Categorical Features: {len(categorical_cols)}
            Missing Values: {missing_cells}
            Duplicates: {duplicate_count}
            """
            
            with st.spinner("Generating AI insights..."):
                insights = generate_text(
                    f"Provide key insights and actionable recommendations for this dataset: {dataset_summary}",
                    dataset_summary
                )
                st.write(insights)
            
            return
        
        # Variable Analysis Function
        def function_question_variable(selected_variable):
            var_stats = df[selected_variable].describe() if df[selected_variable].dtype in ['float64', 'int64'] else df[selected_variable].value_counts()
            missing_count = df[selected_variable].isnull().sum()
            data_type = str(df[selected_variable].dtype)
            
            context = f"Variable: {selected_variable}, Data type: {data_type}, Missing values: {missing_count}, Statistics: {var_stats}"
            
            st.write(f"**Analyzing Variable: {selected_variable}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Type", data_type)
            with col2:
                st.metric("Missing Values", missing_count)
            with col3:
                st.metric("Unique Values", df[selected_variable].nunique())
            
            # Display chart
            if df[selected_variable].dtype in ['float64', 'int64']:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                df[selected_variable].hist(bins=30, ax=ax1)
                ax1.set_title(f'Distribution of {selected_variable}')
                
                df.boxplot(column=selected_variable, ax=ax2)
                ax2.set_title(f'Box Plot of {selected_variable}')
                
                st.pyplot(fig)
                st.line_chart(df[selected_variable])
            else:
                value_counts = df[selected_variable].value_counts()
                st.bar_chart(value_counts)
                st.write("**Top Categories:**")
                st.write(value_counts.head(10))
            
            # AI insights
            summary_statistics = generate_text(f"Provide detailed statistical analysis for the variable '{selected_variable}' with the following data: {context}")
            st.write("**Statistical Analysis:**")
            st.write(summary_statistics)
            
            return
        
        # Custom Question Function
        def function_question_dataframe(user_question):
            dataset_context = f"""
            Dataset Information:
            - Shape: {df.shape[0]} rows, {df.shape[1]} columns
            - Columns: {list(df.columns)}
            - Data types: {dict(df.dtypes)}
            - Missing values per column: {dict(df.isnull().sum())}
            - Numeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}
            - Categorical columns: {df.select_dtypes(include=['object']).columns.tolist()}
            """
            
            enhanced_question = f"""
            User Question: {user_question}
            Dataset Context: {dataset_context}
            Please provide a detailed, specific answer based on the actual data provided.
            """
            
            with st.spinner("Analyzing your data and generating response..."):
                dataframe_info = generate_text(enhanced_question, dataset_context)
                st.write("**AI Analysis Result:**")
                st.write(dataframe_info)
            return
        
        # Enhanced Model Training Function
        def train_model_enhanced(target_col, feature_cols, selected_models, problem_type, test_size, random_state, use_cv, cv_folds):
            
            # Prepare the data
            X = df[feature_cols].select_dtypes(include=[np.number])
            y = df[target_col]
            
            if X.empty:
                st.error("No numeric features selected for training.")
                return
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            if problem_type == "Classification":
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y.fillna('Unknown'))
                    st.info(f"Encoded categorical target. Classes: {list(le.classes_)}")
                else:
                    y = y.fillna(y.mode()[0] if not y.mode().empty else 0)
            else:
                y = y.fillna(y.mean()) if y.dtype in ['float64', 'int64'] else y.fillna(0)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            # Define models
            regression_models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
                'Support Vector Regression': SVR(kernel='rbf'),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=1.0)
            }
            
            classification_models = {
                'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
                'Support Vector Machine': SVC(random_state=random_state),
                'Decision Tree': DecisionTreeClassifier(random_state=random_state)
            }
            
            models_dict = regression_models if problem_type == "Regression" else classification_models
            models_to_train = {name: model for name, model in models_dict.items() if name in selected_models}
            
            st.write(f"**Training {len(models_to_train)} models...**")
            
            results = {}
            progress_bar = st.progress(0)
            
            for i, (name, model) in enumerate(models_to_train.items()):
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    if problem_type == "Regression":
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        results[name] = {'MSE': mse, 'R²': r2}
                    else:
                        accuracy = accuracy_score(y_test, y_pred)
                        results[name] = {'Accuracy': accuracy}
                
                except Exception as e:
                    st.error(f"{name} - Training Error: {str(e)}")
                
                progress_bar.progress((i + 1) / len(models_to_train))
            
            # Display results
            if results:
                st.write("## 📊 Model Performance Results")
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df.round(4))
                
                if problem_type == "Regression":
                    best_model = max(results.keys(), key=lambda x: results[x]['R²'])
                    st.success(f"🏆 Best Model: **{best_model}** with R² = {results[best_model]['R²']:.4f}")
                else:
                    best_model = max(results.keys(), key=lambda x: results[x]['Accuracy'])
                    st.success(f"🏆 Best Model: **{best_model}** with Accuracy = {results[best_model]['Accuracy']:.4f}")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type",
            ["Full Data Analysis", "Variable Analysis", "Custom Question", "Model Training"]
        )
        
        if analysis_type == "Full Data Analysis":
            function_agent()
        
        elif analysis_type == "Variable Analysis":
            user_question_variable = st.selectbox('What variable are you interested in?', df.columns)
            if user_question_variable:
                if st.button("Analyze Variable", key=f"analyze_{user_question_variable}"):
                    function_question_variable(user_question_variable)
        
        elif analysis_type == "Custom Question":
            st.subheader("Ask Any Question About Your Data")
            user_question_dataframe = st.text_input('What do you want to know about your dataframe?', 
                                                   placeholder="e.g., What are the main patterns in this data?")
            
            example_questions = [
                "What are the key insights from this dataset?",
                "Which variables are most correlated?", 
                "What data quality issues should I be aware of?",
                "What machine learning approach would work best?"
            ]
            
            selected_example = st.selectbox("Or choose an example question:", [""] + example_questions)
            
            if selected_example:
                user_question_dataframe = selected_example
            
            if st.button("Get Answer", disabled=not user_question_dataframe) and user_question_dataframe:
                function_question_dataframe(user_question_dataframe)
        
        elif analysis_type == "Model Training":
            st.subheader("🤖 Machine Learning Model Training")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("Need at least 2 numeric columns for model training.")
            else:
                # Problem Type Selection
                problem_type = st.radio("What type of problem?", ["Regression", "Classification"])
                
                # Target Selection
                if problem_type == "Classification":
                    all_cols = numeric_cols + categorical_cols
                    target_col = st.selectbox("Select Target Column", all_cols)
                else:
                    target_col = st.selectbox("Select Target Column", numeric_cols)
                
                # Feature Selection
                available_features = [col for col in numeric_cols if col != target_col]
                feature_cols = st.multiselect("Select Feature Columns", available_features)
                
                # Model Selection
                if problem_type == "Regression":
                    available_models = ["Linear Regression", "Random Forest", "Gradient Boosting", "Support Vector Regression", "Ridge Regression", "Lasso Regression"]
                else:
                    available_models = ["Logistic Regression", "Random Forest", "Support Vector Machine", "Decision Tree"]
                
                selected_models = st.multiselect("Choose Models to Train", available_models, default=available_models[:2])
                
                # Training Configuration
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
                with col2:
                    random_state = st.number_input("Random State", 0, 1000, 42)
                
                cross_validation = st.checkbox("Use Cross-Validation", value=True)
                cv_folds = 5
                
                if st.button("🚀 Train Selected Models") and feature_cols and selected_models:
                    train_model_enhanced(target_col, feature_cols, selected_models, problem_type, test_size/100, random_state, cross_validation, cv_folds)