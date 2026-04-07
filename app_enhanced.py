import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from cryptography.fernet import Fernet
import io
import base64
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
import numpy as np
import google.generativeai as genai
from scipy import stats
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyCR6eP4NmfNAxfwg9iL_WqF7mLzA3hog6Q"
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
        
        fallback_models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        for model_name in fallback_models:
            try:
                return genai.GenerativeModel(model_name)
            except:
                continue
        return None
    except Exception as e:
        st.sidebar.error(f"Model initialization error: {str(e)}")
        return None

# Initialize encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)
def generate_text(prompt, data_context=""):
    """Enhanced AI text generation with better prompts"""
    try:
        model = get_gemini_model()
        if model is None:
            raise Exception("Could not initialize Gemini model")
        
        enhanced_prompt = f"""
        You are an expert data scientist and machine learning engineer. Provide detailed, actionable insights.
        
        Analysis Request: {prompt}
        Data Context: {data_context}
        
        Requirements:
        - Provide specific, data-driven insights
        - Include statistical interpretations
        - Suggest actionable next steps
        - Use technical terminology appropriately
        - Format with clear structure and bullet points
        - Keep response comprehensive but under 200 words
        """
        
        response = model.generate_content(enhanced_prompt)
        return response.text
    
    except Exception as e:
        return f"""
        **Analysis Summary:**
        {prompt}
        
        **Key Insights from Data:**
        {data_context if data_context else "Please refer to the visualizations and statistics above."}
        
        **Recommendations:**
        • Examine data distributions and patterns
        • Check for outliers and anomalies
        • Consider feature engineering opportunities
        • Validate data quality and completeness
        """

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    try:
        return cipher_suite.decrypt(encrypted_data).decode()
    except Exception as e:
        st.error(f"Decryption failed: {e}")
        raise

# Enhanced UI Configuration
st.set_page_config(
    page_title="🤖 Advanced AI Data Science Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title with enhanced styling
st.markdown('<h1 class="main-header">🤖 Advanced AI Data Science Assistant</h1>', unsafe_allow_html=True)
st.markdown("### 🚀 Powered by Gemini AI | Enhanced Analytics | Advanced ML Pipeline")

# Enhanced sidebar
with st.sidebar:
    st.markdown("## 🎯 Navigation Panel")
    st.markdown("---")
    
    # Feature overview
    st.markdown("""
    ### 📊 Available Features:
    - **🔍 Advanced Data Analysis** - 15+ metrics
    - **📈 Smart Visualizations** - Interactive plots  
    - **🤖 AI-Powered Insights** - Gemini integration
    - **⚡ AutoML Pipeline** - 20+ algorithms
    - **🎯 Feature Engineering** - Automated selection
    - **📋 Custom Reports** - Export capabilities
    """)
    
    st.markdown("---")
    st.caption("💡 Upload CSV to unlock all features")
    st.caption("🔒 Data encrypted for security")
    st.caption("⚡ Real-time AI analysis")

# Initialize session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

def clicked(button):
    st.session_state.clicked[button] = True

# Enhanced welcome section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="feature-box">
    <h3>🎯 Welcome to Advanced Data Science</h3>
    <p>Upload your CSV file to unlock powerful AI-driven analytics, automated machine learning, 
    and comprehensive data insights. Our advanced pipeline includes 20+ ML algorithms, 
    automated feature engineering, and intelligent recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Let's Get Started", key="start_btn", help="Click to begin your data science journey"):
        clicked(1)

if st.session_state.clicked[1]:
    st.markdown("## 📁 Data Upload & Security")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        user_csv = st.file_uploader(
            "Upload your CSV file", 
            type="csv",
            help="Supported formats: CSV files up to 200MB"
        )
    
    with col2:
        if user_csv:
            file_size = len(user_csv.getvalue()) / 1024 / 1024
            st.metric("File Size", f"{file_size:.2f} MB")
            st.success("✅ File uploaded successfully!")
    
    if user_csv is not None:
        # Enhanced file processing
        with st.spinner("🔐 Encrypting and processing your data..."):
            file_content = user_csv.read()
            encrypted_file = encrypt_data(file_content.decode('utf-8'))
            st.success("🔒 Data encrypted successfully!")
            
            try:
                decrypted_file = decrypt_data(encrypted_file)
                df = pd.read_csv(io.StringIO(decrypted_file), low_memory=False)
                
                # Data validation
                if df.empty:
                    st.error("❌ Empty dataset detected!")
                    st.stop()
                
                st.success(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
                st.stop()
        # ENHANCED FULL DATA ANALYSIS FUNCTION
        def enhanced_data_analysis():
            st.markdown("# 📊 Advanced Data Analysis Dashboard")
            
            # Enhanced Overview with more metrics
            st.markdown("## 🎯 Dataset Intelligence Report")
            
            # Advanced metrics calculation
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            completeness = ((total_cells - missing_cells) / total_cells) * 100
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Memory usage analysis
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            avg_row_size = memory_usage / df.shape[0] * 1024 if df.shape[0] > 0 else 0
            
            # Data quality score
            duplicate_ratio = df.duplicated().sum() / len(df) * 100
            quality_score = (completeness + (100 - duplicate_ratio)) / 2
            
            # Enhanced metrics display
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{df.shape[0]:,}</h3>
                    <p>Total Rows</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{df.shape[1]}</h3>
                    <p>Total Columns</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{memory_usage:.1f} MB</h3>
                    <p>Memory Usage</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{completeness:.1f}%</h3>
                    <p>Data Completeness</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                quality_color = "🟢" if quality_score >= 85 else "🟡" if quality_score >= 70 else "🔴"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{quality_color} {quality_score:.1f}%</h3>
                    <p>Quality Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Advanced Data Profiling
            st.markdown("## 🔬 Advanced Data Profiling")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Column Type Analysis")
                dtype_info = pd.DataFrame({
                    'Data Type': df.dtypes.value_counts().index,
                    'Count': df.dtypes.value_counts().values,
                    'Percentage': (df.dtypes.value_counts().values / len(df.columns) * 100).round(1)
                })
                st.dataframe(dtype_info, use_container_width=True)
                
                # Interactive pie chart
                fig = px.pie(dtype_info, values='Count', names='Data Type', 
                           title='Data Types Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🎯 Data Characteristics")
                
                # Advanced statistics
                char_data = {
                    'Metric': ['Numeric Columns', 'Categorical Columns', 'DateTime Columns', 
                              'Unique Rows', 'Duplicate Rows', 'Missing Values', 
                              'Average Row Size', 'Data Density'],
                    'Value': [len(numeric_cols), len(categorical_cols), len(datetime_cols),
                             len(df) - df.duplicated().sum(), df.duplicated().sum(), 
                             missing_cells, f"{avg_row_size:.1f} KB", f"{completeness:.1f}%"],
                    'Status': ['✅' if len(numeric_cols) > 0 else '⚠️',
                              '✅' if len(categorical_cols) > 0 else '⚠️',
                              '✅' if len(datetime_cols) > 0 else '➖',
                              '✅', '⚠️' if df.duplicated().sum() > 0 else '✅',
                              '⚠️' if missing_cells > 0 else '✅',
                              '✅', '✅' if completeness > 90 else '⚠️']
                }
                
                char_df = pd.DataFrame(char_data)
                st.dataframe(char_df, use_container_width=True, hide_index=True)
            
            # Enhanced Missing Values Analysis
            if missing_cells > 0:
                st.markdown("## ❌ Advanced Missing Values Analysis")
                
                missing_analysis = df.isnull().sum()
                missing_percent = (missing_analysis / len(df)) * 100
                
                missing_df = pd.DataFrame({
                    'Column': missing_analysis.index,
                    'Missing Count': missing_analysis.values,
                    'Missing %': missing_percent.values,
                    'Severity': ['🔴 Critical' if x > 50 else '🟡 Moderate' if x > 20 else '🟢 Minor' 
                               for x in missing_percent.values],
                    'Recommendation': ['Drop column' if x > 70 else 'Impute carefully' if x > 30 else 'Simple imputation'
                                     for x in missing_percent.values]
                }).sort_values('Missing Count', ascending=False)
                
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)
                
                with col2:
                    # Interactive missing values heatmap
                    fig = px.imshow(df.isnull().astype(int), 
                                  title="Missing Values Heatmap",
                                  color_continuous_scale="Viridis")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced Statistical Analysis
            if numeric_cols:
                st.markdown("## 📈 Advanced Statistical Analysis")
                
                # Comprehensive statistics
                stats_df = df[numeric_cols].describe()
                
                # Additional statistical measures
                additional_stats = pd.DataFrame({
                    'Skewness': df[numeric_cols].skew(),
                    'Kurtosis': df[numeric_cols].kurtosis(),
                    'CV (%)': (df[numeric_cols].std() / df[numeric_cols].mean() * 100),
                    'IQR': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25),
                    'Outliers (IQR)': [len(df[(df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) | 
                                            (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))))]) 
                                     for col in numeric_cols]
                })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Basic Statistics")
                    st.dataframe(stats_df, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 Advanced Metrics")
                    st.dataframe(additional_stats.round(3), use_container_width=True)
                
                # Interactive correlation analysis
                if len(numeric_cols) > 1:
                    st.markdown("### 🔗 Interactive Correlation Analysis")
                    
                    corr_matrix = df[numeric_cols].corr()
                    
                    # Interactive correlation heatmap
                    fig = px.imshow(corr_matrix, 
                                  title="Correlation Matrix",
                                  color_continuous_scale="RdBu",
                                  aspect="auto")
                    fig.update_layout(width=800, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation insights
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.7:
                                high_corr_pairs.append({
                                    'Variable 1': corr_matrix.columns[i],
                                    'Variable 2': corr_matrix.columns[j],
                                    'Correlation': corr_val,
                                    'Strength': 'Very Strong' if abs(corr_val) > 0.9 else 'Strong'
                                })
                    
                    if high_corr_pairs:
                        st.markdown("#### 🚨 High Correlation Alerts")
                        corr_df = pd.DataFrame(high_corr_pairs)
                        st.dataframe(corr_df, use_container_width=True, hide_index=True)
            
            # Enhanced Visualization Suite
            st.markdown("## 🎨 Advanced Visualization Suite")
            
            if numeric_cols:
                # Interactive distribution plots
                st.markdown("### 📊 Interactive Distribution Analysis")
                
                selected_cols = st.multiselect(
                    "Select columns for distribution analysis:",
                    numeric_cols,
                    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
                )
                
                if selected_cols:
                    # Create subplots for distributions
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=[f'Distribution of {col}' for col in selected_cols[:4]],
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    for i, col in enumerate(selected_cols[:4]):
                        row = i // 2 + 1
                        col_pos = i % 2 + 1
                        
                        fig.add_trace(
                            go.Histogram(x=df[col], name=col, nbinsx=30),
                            row=row, col=col_pos
                        )
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # AI-Powered Insights
            st.markdown("## 🤖 AI-Powered Data Insights")
            
            with st.spinner("🧠 Generating AI insights..."):
                context = f"""
                Dataset: {df.shape[0]} rows, {df.shape[1]} columns
                Quality Score: {quality_score:.1f}%
                Numeric Columns: {len(numeric_cols)}
                Categorical Columns: {len(categorical_cols)}
                Missing Values: {missing_cells} ({(missing_cells/total_cells)*100:.1f}%)
                Memory Usage: {memory_usage:.1f} MB
                """
                
                insights = generate_text(
                    "Provide comprehensive data analysis insights and recommendations for this dataset",
                    context
                )
                
                st.markdown("### 💡 Key Insights & Recommendations")
                st.write(insights)
            
            return df