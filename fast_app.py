import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from cryptography.fernet import Fernet
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyCR6eP4NmfNAxfwg9iL_WqF7mLzA3hog6Q"
genai.configure(api_key=GEMINI_API_KEY)

# ── Light theme for all matplotlib charts ──
plt.rcParams.update({
    'figure.facecolor':  '#ffffff',
    'axes.facecolor':    '#f8f9fa',
    'axes.edgecolor':    '#cccccc',
    'axes.labelcolor':   '#333333',
    'xtick.color':       '#555555',
    'ytick.color':       '#555555',
    'text.color':        '#333333',
    'grid.color':        '#e0e0e0',
    'grid.alpha':        0.7,
    'axes.titlecolor':   '#222222',
    'axes.titlesize':    12,
    'axes.labelsize':    10,
    'legend.facecolor':  '#ffffff',
    'legend.edgecolor':  '#cccccc',
    'legend.labelcolor': '#333333',
})

# Test API and get available models
def test_gemini_api():
    try:
        models = genai.list_models()
        available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        st.sidebar.write("Available Gemini Models:")
        for model in available_models[:3]:  # Show first 3
            st.sidebar.write(f"• {model}")
        return available_models
    except Exception as e:
        st.sidebar.error(f"API Test Failed: {str(e)}")
        return []

# Initialize Gemini model
@st.cache_resource
def get_gemini_model():
    try:
        # Get available models and use the first one that supports generateContent
        models = genai.list_models()
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name
                st.sidebar.success(f"Using model: {model_name}")
                return genai.GenerativeModel(model_name)
        
        # Fallback to common model names
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

# Initialize encryption key and cipher (For real usage, load this from a secure location)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def generate_text(prompt, data_context=""):
    """Generate intelligent responses using Gemini API"""
    try:
        model = get_gemini_model()
        
        if model is None:
            raise Exception("Could not initialize Gemini model")
        
        # Enhanced prompt with data science context and uniqueness
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
        # Enhanced fallback with detailed analysis based on context
        error_msg = str(e)
        
        # Only show error once, not repeatedly
        if "429" in error_msg or "quota" in error_msg.lower():
            st.warning("⚠️ AI service rate limit reached. Using enhanced statistical analysis instead.")
        else:
            st.info("ℹ️ Using statistical analysis (AI service unavailable)")
        
        # Provide comprehensive fallback analysis based on prompt type
        if "comprehensive insights" in prompt.lower() or "recommendations" in prompt.lower():
            return f"""
            **📊 Comprehensive Data Analysis (Statistical Mode)**
            
            {data_context}
            
            **Key Observations:**
            • All statistical metrics, visualizations, and tests are fully functional
            • Quality scores and grades are calculated from actual data
            • Feature importance and correlation analysis are data-driven
            • Outlier detection uses multiple validated methods
            
            **Recommended Actions:**
            1. **Data Quality**: Review the quality grade and address any issues flagged
            2. **Statistical Tests**: Check normality tests to validate modeling assumptions
            3. **Feature Selection**: Use importance rankings and correlation analysis
            4. **Outlier Treatment**: Review detected outliers and decide on handling strategy
            5. **Model Selection**: Choose algorithms based on data characteristics shown
            
            **Next Steps:**
            • Use Variable Analysis for detailed column inspection
            • Try Model Training with selected features
            • Export the comprehensive report for documentation
            """
        
        elif "variable" in prompt.lower() or "column" in prompt.lower():
            return f"""
            **📈 Variable Analysis (Statistical Mode)**
            
            {data_context}
            
            **Statistical Insights:**
            • Distribution metrics are calculated from actual data
            • Outliers are detected using validated statistical methods
            • Correlations show real relationships in your data
            
            **Recommendations:**
            • Review the distribution plots for skewness and patterns
            • Check outlier counts and decide if they're valid or errors
            • Examine correlations with other variables
            • Consider transformations if distribution is highly skewed
            """
        
        elif "quality" in prompt.lower():
            return f"""
            **✅ Data Quality Assessment (Statistical Mode)**
            
            {data_context}
            
            **Quality Evaluation:**
            • Completeness, uniqueness, and consistency scores are data-driven
            • Quality grade reflects actual data characteristics
            • All metrics are calculated using standard statistical methods
            
            **Action Items:**
            • Address missing values if completeness < 90%
            • Remove duplicates if uniqueness < 95%
            • Fix consistency issues in categorical data
            • Handle outliers based on domain knowledge
            """
        
        elif "normality" in prompt.lower() or "distribution" in prompt.lower():
            return f"""
            **📊 Distribution Analysis (Statistical Mode)**
            
            {data_context}
            
            **Statistical Tests:**
            • Shapiro-Wilk and Anderson-Darling tests are performed
            • Q-Q plots provide visual normality assessment
            • Skewness and kurtosis indicate distribution shape
            
            **Interpretation:**
            • Normal distributions (p > 0.05): Use parametric tests
            • Non-normal distributions: Consider transformations or non-parametric tests
            • High skewness (|skew| > 1): Log or Box-Cox transformation may help
            """
        
        else:
            return f"""
            **📊 Statistical Analysis (Enhanced Mode)**
            
            {data_context if data_context else "All visualizations and statistical metrics are fully functional."}
            
            **Available Analysis:**
            • ✅ All 15 analysis sections are working
            • ✅ Interactive visualizations with zoom/pan/hover
            • ✅ Statistical tests (normality, independence, etc.)
            • ✅ Feature importance and correlation analysis
            • ✅ Multi-method outlier detection
            • ✅ Quality scoring and grading
            • ✅ Export capabilities
            
            **Note:** AI text generation is temporarily unavailable, but all data-driven analysis, 
            calculations, visualizations, and statistical tests are fully operational.
            """

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Science Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Professional CSS Styling ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

/* ── GLOBAL light theme ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

/* ── Main background: clean light ── */
.stApp { background: #f0f2f6 !important; }

/* ── All text dark by default ── */
.stApp, .stApp *, p, span, div, label, li, h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
.element-container, .element-container * { color: #1a1a2e !important; }

/* ── Metric values ── */
[data-testid="stMetricValue"]  { color: #5b21b6 !important; font-size: 1.8em !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"]  { color: #6b7280 !important; font-size: 0.85em !important; }
[data-testid="stMetricDelta"]  { color: #059669 !important; }

/* ── Headers ── */
.stApp h1, .stApp h2, .stApp h3 { color: #1e1b4b !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%) !important;
    border-right: 2px solid #4f46e5 !important;
}
[data-testid="stSidebar"] * { color: #e0e7ff !important; }
[data-testid="stSidebar"] .stRadio label { color: #e0e7ff !important; }

/* ── Inputs ── */
.stSelectbox > div > div, .stMultiSelect > div > div,
.stTextInput > div > div > input, .stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: #ffffff !important; color: #1a1a2e !important;
    border: 1px solid #c7d2fe !important; border-radius: 8px !important;
}
.stRadio > div label, .stCheckbox > div label { color: #1a1a2e !important; }
.stSlider > div label { color: #1a1a2e !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important;
    border-radius: 25px !important; padding: 10px 30px !important;
    font-weight: 600 !important; transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.35) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(102,126,234,0.5) !important; }

/* ── Info/Warning/Success/Error ── */
.stSuccess, .stSuccess * { background: #d1fae5 !important; border-left: 4px solid #10b981 !important; border-radius: 8px !important; color: #065f46 !important; }
.stWarning, .stWarning * { background: #fef3c7 !important; border-left: 4px solid #f59e0b !important; border-radius: 8px !important; color: #92400e !important; }
.stError,   .stError *   { background: #fee2e2 !important; border-left: 4px solid #ef4444 !important; border-radius: 8px !important; color: #991b1b !important; }
.stInfo,    .stInfo *    { background: #dbeafe !important; border-left: 4px solid #3b82f6 !important; border-radius: 8px !important; color: #1e40af !important; }

/* ── Dataframes ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] * { color: #1a1a2e !important; }

/* ── Expander ── */
.streamlit-expanderHeader { color: #1e1b4b !important; background: #e0e7ff !important; border-radius: 8px !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 40px 30px; border-radius: 20px; text-align: center;
    margin-bottom: 30px; box-shadow: 0 20px 60px rgba(102,126,234,0.35);
}
.hero-banner h1 { font-size: 3em; font-weight: 700; color: white !important; margin: 0; text-shadow: 2px 2px 10px rgba(0,0,0,0.2); }
.hero-banner p  { font-size: 1.2em; color: rgba(255,255,255,0.92) !important; margin-top: 10px; }

/* ── Feature Cards ── */
.feature-card {
    background: #ffffff; border: 1px solid #e0e7ff;
    border-radius: 15px; padding: 20px; text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 2px 10px rgba(102,126,234,0.1);
}
.feature-card:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(102,126,234,0.25); }
.feature-card h3 { color: #5b21b6 !important; font-size: 1.05em; margin: 10px 0 5px 0; }
.feature-card p  { color: #6b7280 !important; font-size: 0.85em; margin: 0; }
.feature-icon    { font-size: 2.5em; }

/* ── Animated Section Headers ── */
@keyframes slideIn { from { opacity:0; transform:translateX(-20px); } to { opacity:1; transform:translateX(0); } }
.section-title {
    animation: slideIn 0.4s ease forwards;
    background: linear-gradient(90deg, #e0e7ff, #f5f3ff);
    border-left: 4px solid #667eea;
    padding: 12px 20px; border-radius: 0 10px 10px 0;
    margin: 25px 0 15px 0; font-size: 1.2em; font-weight: 700;
    color: #1e1b4b !important; display: flex; align-items: center; gap: 10px;
}

/* ── Grade Cards ── */
.grade-A { background: linear-gradient(135deg, #10b981, #059669); border-radius: 15px; padding: 20px; text-align: center; color: white !important; }
.grade-B { background: linear-gradient(135deg, #3b82f6, #2563eb); border-radius: 15px; padding: 20px; text-align: center; color: white !important; }
.grade-C { background: linear-gradient(135deg, #f59e0b, #d97706); border-radius: 15px; padding: 20px; text-align: center; color: white !important; }
.grade-D { background: linear-gradient(135deg, #ef4444, #dc2626); border-radius: 15px; padding: 20px; text-align: center; color: white !important; }
.grade-letter { font-size: 4em; font-weight: 700; color: white !important; }
.grade-text   { font-size: 1em; margin-top: 5px; color: white !important; }

/* ── Upload Area ── */
.upload-area {
    background: #f5f3ff; border: 2px dashed #a78bfa;
    border-radius: 15px; padding: 30px; text-align: center; margin: 20px 0;
}
.upload-area * { color: #5b21b6 !important; }

/* ── Nav items (sidebar) ── */
.nav-item {
    background: rgba(255,255,255,0.1); border-left: 3px solid #a5b4fc;
    padding: 7px 14px; margin: 4px 0; border-radius: 0 8px 8px 0; font-size: 0.88em;
    color: #e0e7ff !important;
}

/* ── Chat Interface ── */
.chat-container {
    background: #f9fafb; border: 1px solid #e0e7ff;
    border-radius: 15px; padding: 20px; max-height: 500px; overflow-y: auto; margin-bottom: 15px;
}
.chat-msg-user {
    background: linear-gradient(135deg, #667eea, #764ba2); color: white !important;
    padding: 12px 18px; border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 20%; font-size: 0.95em; box-shadow: 0 4px 12px rgba(102,126,234,0.3);
}
.chat-msg-ai {
    background: #ffffff; border: 1px solid #e0e7ff; color: #1a1a2e !important;
    padding: 12px 18px; border-radius: 18px 18px 18px 4px; margin: 8px 20% 8px 0; font-size: 0.95em;
}
.chat-msg-ai .ai-label { color: #5b21b6 !important; font-weight: 700; font-size: 0.8em; margin-bottom: 5px; }

/* ── AI Response Box ── */
.ai-response-box {
    background: #f5f3ff; border: 1px solid #c4b5fd;
    border-radius: 12px; padding: 25px 20px 20px 20px; margin: 15px 0; position: relative;
}
.ai-response-box, .ai-response-box * { color: #1e1b4b !important; }
.ai-response-box::before {
    content: "🤖 AI Insights"; position: absolute; top: -12px; left: 15px;
    background: linear-gradient(135deg, #667eea, #764ba2); color: white !important;
    padding: 3px 12px; border-radius: 10px; font-size: 0.75em; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ─── UI Helper Functions ─────────────────────────────────────────────────────
def section_header(icon, title):
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)

def ai_response(text):
    st.markdown(f'<div class="ai-response-box">{text}</div>', unsafe_allow_html=True)

def gauge_chart(value, title, color="#667eea"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"color": "#333333", "size": 13}},
        number={"suffix": "%", "font": {"color": "#222222", "size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#666666", "tickfont": {"color": "#666666", "size": 10}},
            "bar": {"color": color},
            "bgcolor": "#f0f0f0",
            "borderwidth": 1,
            "bordercolor": "#cccccc",
            "steps": [
                {"range": [0, 50],  "color": "rgba(214,48,49,0.15)"},
                {"range": [50, 75], "color": "rgba(253,203,110,0.2)"},
                {"range": [75, 100],"color": "rgba(0,184,148,0.15)"},
            ],
            "threshold": {"line": {"color": "#333333", "width": 2}, "thickness": 0.75, "value": value}
        }
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=180,
        margin=dict(l=15, r=15, t=35, b=5),
        font={"color": "#333333"}
    )
    return fig

# ─── Encryption / Decryption ─────────────────────────────────────────────────
def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    try:
        return cipher_suite.decrypt(encrypted_data).decode()
    except Exception as e:
        st.error(f"Decryption failed: {e}")
        raise

# ─── Hero Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🤖 AI Data Science Assistant</h1>
    <p>Advanced ML • Interactive Analytics • AI-Powered Insights</p>
</div>
""", unsafe_allow_html=True)

# ─── Feature Cards ───────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
features = [
    ("📊", "15 Analysis Sections", "Comprehensive data profiling"),
    ("🤖", "AI-Powered Insights", "Smart automated analysis"),
    ("🧠", "10+ ML Algorithms", "Auto model comparison"),
    ("🔒", "Data Encryption", "Secure file processing"),
    ("📥", "Export Reports", "Download CSV summaries"),
]
for col, (icon, title, desc) in zip([col1,col2,col3,col4,col5], features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── AI Status ───────────────────────────────────────────────────────────────
try:
    test_model = get_gemini_model()
    if test_model:
        st.success("✅ AI service connected — Full AI insights available")
    else:
        st.info("ℹ️ AI service unavailable — Using enhanced statistical analysis mode")
except:
    st.info("ℹ️ AI service unavailable — Using enhanced statistical analysis mode")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 15px 0;'>
        <div style='font-size:2.5em;'>🤖</div>
        <div style='font-size:1.2em; font-weight:700; color:#667eea;'>AI Data Science</div>
        <div style='font-size:0.8em; color:#aaa;'>Assistant v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Analysis Mode Selector (always visible) ──
    st.markdown("**🧭 Choose Analysis Mode:**")
    analysis_type = st.radio(
        "Analysis Mode",
        ["📊 Full Data Analysis", "🔬 Variable Analysis", "💬 Custom Question", "🤖 Model Training"],
        label_visibility="collapsed"
    )
    # Strip emoji prefix for logic
    analysis_type = analysis_type.split(" ", 1)[1]

    st.divider()

    st.markdown("**📋 How to Use:**")
    steps = [("1","Upload CSV"), ("2","Choose Mode"), ("3","Explore Results"), ("4","Export Report")]
    for num, step in steps:
        st.markdown(f"""
        <div class="nav-item">
            <span style='background:linear-gradient(135deg,#667eea,#764ba2);color:white;
            border-radius:50%;padding:2px 8px;font-size:0.8em;margin-right:8px;'>{num}</span>
            {step}
        </div>""", unsafe_allow_html=True)

    st.divider()
    if analysis_type == "Full Data Analysis":
        st.markdown("**🔍 Full Analysis (15 Sections):**")
        sections = ["Dataset Overview","Data Quality","Statistics","ML Readiness",
                    "Normality Tests","Feature Importance","Outlier Detection",
                    "Time Series","Categorical Analysis","Distribution","Correlation Network",
                    "AI Insights","Data Explorer","Quality Report","Export"]
        for i, s in enumerate(sections, 1):
            st.markdown(f"<div class='nav-item'>📌 {i}. {s}</div>", unsafe_allow_html=True)
    elif analysis_type == "Variable Analysis":
        st.markdown("**🔬 Variable Analysis:**")
        st.markdown("<div class='nav-item'>Select any column to get deep statistical insights, distribution plots, outlier analysis, and AI commentary.</div>", unsafe_allow_html=True)
    elif analysis_type == "Custom Question":
        st.markdown("**💬 AI Chat:**")
        st.markdown("<div class='nav-item'>Ask anything about your dataset in natural language. Powered by Google Gemini AI.</div>", unsafe_allow_html=True)
    elif analysis_type == "Model Training":
        st.markdown("**🤖 Model Training:**")
        steps_ml = ["Select problem type","Choose target column","Pick features","Select models","Train & compare"]
        for i, s in enumerate(steps_ml, 1):
            st.markdown(f"<div class='nav-item'>⚙️ {i}. {s}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='text-align:center; color:#667eea; font-size:0.8em;'>Made with ❤️ for Capstone 2025</p>", unsafe_allow_html=True)

# Initialize the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True

st.markdown("""
<div style='text-align:center; margin: 30px 0 10px 0;'>
    <div style='font-size:1.1em; color:#aaa; margin-bottom:15px;'>Ready to unlock insights from your data?</div>
</div>""", unsafe_allow_html=True)

col_btn = st.columns([2,1,2])[1]
with col_btn:
    st.button("🚀 Let's Get Started", on_click=clicked, args=[1], use_container_width=True)

if st.session_state.clicked[1]:
    st.markdown("""
    <div class="upload-area">
        <div style='font-size:2.5em;'>📂</div>
        <div style='font-size:1.1em; color:#e0e0e0; margin:10px 0 5px 0;'>Upload your CSV dataset</div>
        <div style='font-size:0.85em; color:#aaa;'>Drag & drop or click to browse</div>
    </div>""", unsafe_allow_html=True)
    user_csv = st.file_uploader("Upload CSV", type="csv", label_visibility="hidden")
    
    if user_csv is not None:
        # Encrypt the file for demonstration (you can test decryption with a sample encrypted file)
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
        
        # Function sidebar
        @st.cache_data
        def steps_eda():
            steps_eda = generate_text('What are the steps of EDA?')
            return steps_eda
        
        # Enhanced Full Data Analysis Function with Advanced Features
        def function_agent():
            from scipy import stats
            from scipy.stats import shapiro, normaltest, anderson, kstest, chi2_contingency
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            st.markdown("""
            <div style='background:linear-gradient(135deg,rgba(102,126,234,0.2),rgba(118,75,162,0.2));
                        border:1px solid rgba(102,126,234,0.3); border-radius:15px; padding:25px;
                        text-align:center; margin-bottom:25px;'>
                <div style='font-size:2em; font-weight:700; color:#e0e0ff;'>📊 Comprehensive Data Analysis Report</div>
                <div style='color:#aaa; margin-top:8px;'>AI-powered insights • 15 analysis sections • Interactive visualizations</div>
            </div>""", unsafe_allow_html=True)
            
            # === DATASET OVERVIEW ===
            section_header("📋", "1. Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            with col4:
                st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            
            # Data types breakdown
            st.subheader("📊 Data Types Distribution")
            dtype_counts = df.dtypes.value_counts()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Column Types:**")
                for dtype, count in dtype_counts.items():
                    st.write(f"• {dtype}: {count} columns")
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                dtype_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title('Data Types Distribution')
                ax.set_xlabel('Data Type')
                ax.set_ylabel('Number of Columns')
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            # Sample data
            st.subheader("👀 Sample Data")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**First 5 rows:**")
                st.dataframe(df.head())
            with col2:
                st.write("**Last 5 rows:**")
                st.dataframe(df.tail())
            
            # === DATA QUALITY EVALUATION METRICS ===
            section_header("🎯", "2. Data Quality Evaluation Metrics")
            
            # Calculate comprehensive data quality metrics
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # Memory usage in MB
            completeness_score = ((total_cells - missing_cells) / total_cells) * 100
            
            duplicate_count = df.duplicated().sum()
            uniqueness_score = ((df.shape[0] - duplicate_count) / df.shape[0]) * 100
            
            # Data consistency metrics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Calculate consistency score based on data types
            consistency_issues = 0
            for col in categorical_cols:
                # Check for mixed case, extra spaces, etc.
                if df[col].dtype == 'object':
                    unique_vals = df[col].dropna().astype(str)
                    cleaned_vals = unique_vals.str.strip().str.lower()
                    if len(unique_vals.unique()) != len(cleaned_vals.unique()):
                        consistency_issues += 1
            
            consistency_score = ((len(categorical_cols) - consistency_issues) / max(len(categorical_cols), 1)) * 100
            
            # Overall data quality score
            overall_quality = (completeness_score + uniqueness_score + consistency_score) / 3
            
            # Display quality metrics with gauge charts
            section_header("📈", "Data Quality Scorecard")
            col1, col2, col3, col4 = st.columns(4)
            gauge_colors = {
                "completeness": "#00b894" if completeness_score >= 85 else "#fdcb6e" if completeness_score >= 70 else "#d63031",
                "uniqueness":   "#00b894" if uniqueness_score >= 90 else "#fdcb6e" if uniqueness_score >= 75 else "#d63031",
                "consistency":  "#00b894" if consistency_score >= 85 else "#fdcb6e" if consistency_score >= 70 else "#d63031",
                "overall":      "#00b894" if overall_quality >= 85 else "#fdcb6e" if overall_quality >= 70 else "#d63031",
            }
            with col1:
                st.plotly_chart(gauge_chart(completeness_score, "Completeness", gauge_colors["completeness"]), width='stretch')
            with col2:
                st.plotly_chart(gauge_chart(uniqueness_score, "Uniqueness", gauge_colors["uniqueness"]), width='stretch')
            with col3:
                st.plotly_chart(gauge_chart(consistency_score, "Consistency", gauge_colors["consistency"]), width='stretch')
            with col4:
                st.plotly_chart(gauge_chart(overall_quality, "Overall Quality", gauge_colors["overall"]), width='stretch')
            
            # Grade badge
            grade = "A" if overall_quality >= 90 else "B" if overall_quality >= 75 else "C" if overall_quality >= 60 else "D"
            grade_label = {"A": "Excellent", "B": "Good", "C": "Fair", "D": "Needs Work"}
            st.markdown(f"""
            <div class="grade-{grade}" style="max-width:200px; margin: 10px auto;">
                <div class="grade-letter">{grade}</div>
                <div class="grade-text">{grade_label[grade]}</div>
            </div>""", unsafe_allow_html=True)
            
            # Missing values analysis with evaluation
            section_header("❌", "Missing Values Analysis & Evaluation")
            missing_data = df.isnull().sum()
            missing_percent = (missing_data / len(df)) * 100
            
            if missing_data.sum() > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing Percentage': missing_percent.values,
                    'Severity': ['🔴 Critical' if x > 50 else '🟡 Moderate' if x > 20 else '🟢 Minor' for x in missing_percent.values]
                }).sort_values('Missing Count', ascending=False)
                
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(missing_df)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(df.isnull(), cbar=True, cmap='viridis', ax=ax)
                    ax.set_title('Missing Values Heatmap')
                    st.pyplot(fig)
                
                # AI insights on missing data
                missing_context = f"Missing data analysis: {missing_df.to_dict()}"
                missing_insights = generate_text("Analyze the missing data patterns and suggest handling strategies", missing_context)
                st.write("**🤖 AI Insights on Missing Data:**")
                st.write(missing_insights)
            else:
                st.success("✅ No missing values found in the dataset!")
            
            # Duplicate analysis with evaluation
            st.subheader("🔄 Duplicate Records Analysis & Evaluation")
            duplicates = df.duplicated()
            duplicate_count = duplicates.sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Duplicate Rows", f"{duplicate_count:,}")
                st.metric("Duplicate Percentage", f"{(duplicate_count/len(df)*100):.2f}%")
                
                # Duplicate severity assessment
                dup_severity = "🔴 High" if duplicate_count/len(df) > 0.1 else "🟡 Medium" if duplicate_count/len(df) > 0.05 else "🟢 Low"
                st.write(f"**Severity Level:** {dup_severity}")
            
            with col2:
                if duplicate_count > 0:
                    st.write("**Sample Duplicate Records:**")
                    st.dataframe(df[duplicates].head())
                else:
                    st.success("✅ No duplicate records found!")
            
            # === STATISTICAL ANALYSIS WITH EVALUATION METRICS ===
            section_header("📈", "3. Statistical Analysis & Evaluation")
            
            if numeric_cols:
                st.subheader("🔢 Numeric Variables Evaluation")
                
                # Enhanced statistics with evaluation
                numeric_stats = df[numeric_cols].describe()
                st.dataframe(numeric_stats)
                
                # Statistical evaluation metrics
                st.write("**📊 Statistical Evaluation Metrics:**")
                eval_metrics = pd.DataFrame({
                    'Skewness': df[numeric_cols].skew(),
                    'Kurtosis': df[numeric_cols].kurtosis(),
                    'Coefficient of Variation': (df[numeric_cols].std() / df[numeric_cols].mean()) * 100,
                    'Normality (Shapiro p-value)': [stats.shapiro(df[col].dropna().sample(min(5000, len(df[col].dropna()))))[1] if len(df[col].dropna()) > 3 else np.nan for col in numeric_cols],
                    'Outlier Count (IQR method)': [len(df[(df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) | 
                                                        (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))))]) for col in numeric_cols]
                })
                st.dataframe(eval_metrics.round(4))
                
                # Distribution quality assessment
                st.write("**📈 Distribution Quality Assessment:**")
                for col in numeric_cols[:3]:  # Limit to first 3 for performance
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        skew_val = abs(col_data.skew())
                        kurt_val = abs(col_data.kurtosis())
                        
                        dist_quality = "🟢 Normal" if skew_val < 0.5 and kurt_val < 3 else "🟡 Moderate Skew" if skew_val < 2 else "🔴 Highly Skewed"
                        st.write(f"• **{col}**: {dist_quality} (Skew: {skew_val:.2f}, Kurtosis: {kurt_val:.2f})")
                
                # Correlation analysis with evaluation
                if len(numeric_cols) > 1:
                    st.subheader("🔗 Correlation Analysis & Multicollinearity Evaluation")
                    corr_matrix = df[numeric_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, ax=ax)
                    ax.set_title('Correlation Matrix')
                    st.pyplot(fig)
                    
                    # Multicollinearity evaluation
                    high_corr_pairs = []
                    moderate_corr_pairs = []
                    
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = abs(corr_matrix.iloc[i, j])
                            if corr_val > 0.8:
                                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                            elif corr_val > 0.6:
                                moderate_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                    
                    st.write("**🚨 Multicollinearity Assessment:**")
                    if high_corr_pairs:
                        st.write("**High Correlations (>0.8) - Potential Multicollinearity:**")
                        for col1, col2, corr in high_corr_pairs:
                            st.write(f"• 🔴 {col1} ↔ {col2}: {corr:.3f}")
                    
                    if moderate_corr_pairs:
                        st.write("**Moderate Correlations (0.6-0.8):**")
                        for col1, col2, corr in moderate_corr_pairs:
                            st.write(f"• 🟡 {col1} ↔ {col2}: {corr:.3f}")
                    
                    if not high_corr_pairs and not moderate_corr_pairs:
                        st.success("✅ No significant multicollinearity detected!")
            
            # Categorical analysis with evaluation
            if categorical_cols:
                st.subheader("📝 Categorical Variables Evaluation")
                
                cat_eval_metrics = []
                for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                    unique_count = df[col].nunique()
                    mode_freq = df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                    mode_percentage = (mode_freq / len(df)) * 100
                    
                    # Calculate entropy for diversity measure
                    value_counts = df[col].value_counts()
                    probabilities = value_counts / len(df)
                    entropy = -sum(probabilities * np.log2(probabilities + 1e-10))
                    
                    cat_eval_metrics.append({
                        'Column': col,
                        'Unique Values': unique_count,
                        'Cardinality': 'High' if unique_count > len(df) * 0.5 else 'Medium' if unique_count > 10 else 'Low',
                        'Mode Frequency': mode_freq,
                        'Mode Percentage': f"{mode_percentage:.1f}%",
                        'Entropy (Diversity)': f"{entropy:.2f}",
                        'Balance': 'Balanced' if mode_percentage < 50 else 'Imbalanced'
                    })
                
                cat_eval_df = pd.DataFrame(cat_eval_metrics)
                st.dataframe(cat_eval_df)
                
                # Visualize top categorical variables
                for col in categorical_cols[:3]:
                    st.write(f"**Distribution of {col}:**")
                    value_counts = df[col].value_counts().head(10)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    value_counts.plot(kind='bar', ax=ax, color='lightcoral')
                    ax.set_title(f'Top 10 Values in {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            
            # === ADVANCED EVALUATION METRICS ===
            section_header("🎯", "4. Advanced Evaluation Metrics")
            
            # Data dimensionality evaluation
            st.subheader("📐 Dimensionality Analysis")
            
            if len(numeric_cols) > 2:
                # PCA for dimensionality assessment
                numeric_data = df[numeric_cols].fillna(df[numeric_cols].mean())
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_data)
                
                pca = PCA()
                pca.fit(scaled_data)
                
                # Calculate cumulative explained variance
                cumsum_var = np.cumsum(pca.explained_variance_ratio_)
                n_components_90 = np.argmax(cumsum_var >= 0.9) + 1
                n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Dimensions", len(numeric_cols))
                with col2:
                    st.metric("90% Variance", f"{n_components_90} components")
                with col3:
                    st.metric("95% Variance", f"{n_components_95} components")
                
                # Plot explained variance
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
                ax.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
                ax.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
                ax.set_xlabel('Number of Components')
                ax.set_ylabel('Cumulative Explained Variance')
                ax.set_title('PCA - Cumulative Explained Variance')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            # Data readiness score
            st.subheader("🏆 ML Readiness Assessment")
            
            # Calculate ML readiness factors
            factors = {
                'Data Completeness': completeness_score,
                'Data Uniqueness': uniqueness_score,
                'Feature Diversity': len(numeric_cols) / max(df.shape[1], 1) * 100,
                'Sample Size Adequacy': min(100, (df.shape[0] / max(df.shape[1] * 10, 100)) * 100),
                'Correlation Health': 100 - (len(high_corr_pairs) * 20) if 'high_corr_pairs' in locals() else 100
            }
            
            ml_readiness = sum(factors.values()) / len(factors)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                for factor, score in factors.items():
                    st.metric(factor, f"{score:.1f}%")
                
                readiness_level = "🟢 Ready" if ml_readiness >= 80 else "🟡 Needs Work" if ml_readiness >= 60 else "🔴 Not Ready"
                st.metric("**ML Readiness**", f"{readiness_level} ({ml_readiness:.1f}%)")
            
            with col2:
                # Radar chart for ML readiness
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                
                angles = np.linspace(0, 2 * np.pi, len(factors), endpoint=False)
                values = list(factors.values())
                
                ax.plot(angles, values, 'o-', linewidth=2, color='blue')
                ax.fill(angles, values, alpha=0.25, color='blue')
                ax.set_xticks(angles)
                ax.set_xticklabels(factors.keys())
                ax.set_ylim(0, 100)
                ax.set_title('ML Readiness Radar Chart', pad=20)
                ax.grid(True)
                
                st.pyplot(fig)
            
            # === ADVANCED STATISTICAL TESTS ===
            section_header("🔬", "5. Advanced Statistical Tests")
            
            if numeric_cols:
                st.subheader("📊 Normality Tests")
                
                normality_results = []
                for col in numeric_cols[:10]:  # Limit to first 10 for performance
                    col_data = df[col].dropna()
                    if len(col_data) > 3:
                        # Shapiro-Wilk test
                        shapiro_stat, shapiro_p = shapiro(col_data.sample(min(5000, len(col_data))))
                        
                        # Anderson-Darling test
                        anderson_result = anderson(col_data.sample(min(5000, len(col_data))))
                        
                        normality_results.append({
                            'Variable': col,
                            'Shapiro-Wilk p-value': shapiro_p,
                            'Normal?': '✅ Yes' if shapiro_p > 0.05 else '❌ No',
                            'Anderson Statistic': anderson_result.statistic,
                            'Skewness': col_data.skew(),
                            'Kurtosis': col_data.kurtosis()
                        })
                
                if normality_results:
                    norm_df = pd.DataFrame(normality_results)
                    st.dataframe(norm_df.round(4), width='stretch')
                    
                    # Visualization of normality
                    st.write("**Q-Q Plots for Normality Assessment:**")
                    n_cols_to_plot = min(4, len(numeric_cols))
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    axes = axes.flatten()
                    
                    for i, col in enumerate(numeric_cols[:n_cols_to_plot]):
                        col_data = df[col].dropna()
                        stats.probplot(col_data, dist="norm", plot=axes[i])
                        axes[i].set_title(f'Q-Q Plot: {col}')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # === FEATURE IMPORTANCE ANALYSIS ===
            section_header("🎯", "6. Feature Importance Analysis")
            
            if len(numeric_cols) > 2:
                st.subheader("🔍 Automated Feature Importance")
                
                # Let user select target for importance analysis
                target_for_importance = st.selectbox(
                    "Select target variable for feature importance:",
                    numeric_cols,
                    key="importance_target"
                )
                
                if target_for_importance:
                    features_for_importance = [col for col in numeric_cols if col != target_for_importance]
                    
                    if len(features_for_importance) > 0:
                        X_importance = df[features_for_importance].fillna(df[features_for_importance].mean())
                        y_importance = df[target_for_importance].fillna(df[target_for_importance].mean())
                        
                        # Random Forest Feature Importance
                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf_model.fit(X_importance, y_importance)
                        
                        importance_df = pd.DataFrame({
                            'Feature': features_for_importance,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.dataframe(importance_df.round(4), width='stretch')
                        
                        with col2:
                            fig = px.bar(importance_df, x='Importance', y='Feature', 
                                       orientation='h', title=f'Feature Importance for {target_for_importance}')
                            st.plotly_chart(fig, width='stretch')
                        
                        # Mutual Information
                        mi_scores = mutual_info_regression(X_importance, y_importance, random_state=42)
                        mi_df = pd.DataFrame({
                            'Feature': features_for_importance,
                            'Mutual Information': mi_scores
                        }).sort_values('Mutual Information', ascending=False)
                        
                        st.write("**Mutual Information Scores:**")
                        st.dataframe(mi_df.round(4), width='stretch')
            
            # === OUTLIER DETECTION METHODS ===
            section_header("🚨", "7. Advanced Outlier Detection")
            
            if numeric_cols:
                st.subheader("📍 Multiple Outlier Detection Methods")
                
                outlier_summary = []
                
                for col in numeric_cols[:8]:  # Limit to first 8 columns
                    col_data = df[col].dropna()
                    
                    # IQR Method
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    iqr_outliers = len(col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)])
                    
                    # Z-Score Method
                    z_scores = np.abs(stats.zscore(col_data))
                    z_outliers = len(col_data[z_scores > 3])
                    
                    # Modified Z-Score (using MAD)
                    median = col_data.median()
                    mad = np.median(np.abs(col_data - median))
                    if mad != 0:
                        modified_z_scores = 0.6745 * (col_data - median) / mad
                        mad_outliers = len(col_data[np.abs(modified_z_scores) > 3.5])
                    else:
                        mad_outliers = 0
                    
                    outlier_summary.append({
                        'Variable': col,
                        'IQR Outliers': iqr_outliers,
                        'Z-Score Outliers': z_outliers,
                        'MAD Outliers': mad_outliers,
                        'Total Values': len(col_data),
                        'Outlier %': f"{(iqr_outliers/len(col_data)*100):.2f}%" if len(col_data) > 0 else "0.00%"
                    })
                
                outlier_df = pd.DataFrame(outlier_summary)
                st.dataframe(outlier_df, width='stretch')
                
                # Interactive outlier visualization
                st.write("**Interactive Outlier Visualization:**")
                selected_outlier_col = st.selectbox("Select column for detailed outlier view:", numeric_cols)
                
                if selected_outlier_col:
                    fig = go.Figure()
                    fig.add_trace(go.Box(y=df[selected_outlier_col], name=selected_outlier_col, boxmean='sd'))
                    fig.update_layout(title=f'Box Plot with Outliers: {selected_outlier_col}', height=400)
                    st.plotly_chart(fig, width='stretch')
            
            # === TIME SERIES ANALYSIS (if date columns exist) ===
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Try to detect date columns
            potential_date_cols = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
                    try:
                        pd.to_datetime(df[col])
                        potential_date_cols.append(col)
                    except:
                        pass
            
            if datetime_cols or potential_date_cols:
                section_header("📅", "8. Time Series Analysis")
                
                date_col = datetime_cols[0] if datetime_cols else potential_date_cols[0]
                st.write(f"**Detected time column: {date_col}**")
                
                # Convert to datetime if needed
                if date_col not in datetime_cols:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Time series metrics
                if numeric_cols:
                    ts_col = st.selectbox("Select numeric column for time series analysis:", numeric_cols)
                    
                    if ts_col:
                        ts_data = df[[date_col, ts_col]].dropna().sort_values(date_col)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Time Range", f"{ts_data[date_col].min().date()} to {ts_data[date_col].max().date()}")
                        with col2:
                            st.metric("Data Points", len(ts_data))
                        with col3:
                            trend = "📈 Increasing" if ts_data[ts_col].iloc[-1] > ts_data[ts_col].iloc[0] else "📉 Decreasing"
                            st.metric("Trend", trend)
                        
                        # Interactive time series plot
                        fig = px.line(ts_data, x=date_col, y=ts_col, title=f'{ts_col} Over Time')
                        fig.update_traces(line_color='#1f77b4', line_width=2)
                        st.plotly_chart(fig, width='stretch')
                        
                        # Rolling statistics
                        st.write("**Rolling Statistics:**")
                        window = st.slider("Select rolling window size:", 2, 30, 7)
                        
                        ts_data['Rolling_Mean'] = ts_data[ts_col].rolling(window=window).mean()
                        ts_data['Rolling_Std'] = ts_data[ts_col].rolling(window=window).std()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=ts_data[date_col], y=ts_data[ts_col], 
                                               mode='lines', name='Original', line=dict(color='lightblue')))
                        fig.add_trace(go.Scatter(x=ts_data[date_col], y=ts_data['Rolling_Mean'], 
                                               mode='lines', name=f'{window}-Period MA', line=dict(color='red', width=2)))
                        fig.update_layout(title=f'{ts_col} with {window}-Period Moving Average', height=400)
                        st.plotly_chart(fig, width='stretch')
            
            # === CATEGORICAL VARIABLE ANALYSIS ===
            if categorical_cols:
                section_header("📝", "9. Advanced Categorical Analysis")
                
                st.subheader("🔍 Chi-Square Independence Tests")
                
                if len(categorical_cols) >= 2:
                    cat_col1 = st.selectbox("Select first categorical variable:", categorical_cols, key="cat1")
                    cat_col2 = st.selectbox("Select second categorical variable:", 
                                          [c for c in categorical_cols if c != cat_col1], key="cat2")
                    
                    if cat_col1 and cat_col2:
                        # Create contingency table
                        contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
                        
                        # Chi-square test
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Chi-Square Statistic", f"{chi2:.4f}")
                        with col2:
                            st.metric("P-Value", f"{p_value:.4f}")
                        with col3:
                            independence = "✅ Independent" if p_value > 0.05 else "❌ Dependent"
                            st.metric("Variables are:", independence)
                        
                        # Heatmap of contingency table
                        fig = px.imshow(contingency_table, 
                                      title=f'Contingency Table: {cat_col1} vs {cat_col2}',
                                      labels=dict(x=cat_col2, y=cat_col1, color="Count"),
                                      color_continuous_scale='Blues')
                        st.plotly_chart(fig, width='stretch')
            
            # === DATA DISTRIBUTION COMPARISON ===
            section_header("📊", "10. Distribution Comparison & Analysis")
            
            if len(numeric_cols) >= 2:
                st.subheader("🔄 Compare Distributions")
                
                compare_cols = st.multiselect(
                    "Select columns to compare distributions:",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                )
                
                if len(compare_cols) >= 2:
                    # Violin plots for comparison
                    fig = go.Figure()
                    for col in compare_cols:
                        fig.add_trace(go.Violin(y=df[col], name=col, box_visible=True, meanline_visible=True))
                    
                    fig.update_layout(title='Distribution Comparison (Violin Plots)', height=500)
                    st.plotly_chart(fig, width='stretch')
                    
                    # KS test for distribution similarity
                    st.write("**Kolmogorov-Smirnov Tests (Distribution Similarity):**")
                    ks_results = []
                    
                    for i in range(len(compare_cols)):
                        for j in range(i+1, len(compare_cols)):
                            col1_data = df[compare_cols[i]].dropna()
                            col2_data = df[compare_cols[j]].dropna()
                            
                            ks_stat, ks_p = kstest(col1_data, col2_data)
                            
                            ks_results.append({
                                'Variable 1': compare_cols[i],
                                'Variable 2': compare_cols[j],
                                'KS Statistic': ks_stat,
                                'P-Value': ks_p,
                                'Similar?': '✅ Yes' if ks_p > 0.05 else '❌ No'
                            })
                    
                    if ks_results:
                        ks_df = pd.DataFrame(ks_results)
                        st.dataframe(ks_df.round(4), width='stretch')
            
            # === CORRELATION NETWORK ===
            if len(numeric_cols) > 2:
                section_header("🕸️", "11. Correlation Network Analysis")
                
                corr_threshold = st.slider("Correlation threshold for network:", 0.3, 0.9, 0.6, 0.1)
                
                corr_matrix = df[numeric_cols].corr()
                
                # Create network data
                edges = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) >= corr_threshold:
                            edges.append({
                                'Source': corr_matrix.columns[i],
                                'Target': corr_matrix.columns[j],
                                'Correlation': corr_val,
                                'Strength': 'Strong' if abs(corr_val) > 0.8 else 'Moderate'
                            })
                
                if edges:
                    edge_df = pd.DataFrame(edges)
                    st.write(f"**Found {len(edges)} significant correlations (|r| ≥ {corr_threshold}):**")
                    st.dataframe(edge_df.round(3), width='stretch')
                else:
                    st.info(f"No correlations found above threshold {corr_threshold}")
            
            # === AI INSIGHTS AND RECOMMENDATIONS ===
            section_header("🤖", "12. AI Insights & Recommendations")
            
            # Generate comprehensive insights
            analysis_context = f"""
            Dataset Analysis Summary:
            - Shape: {df.shape}
            - Data Quality Score: {overall_quality:.1f}%
            - ML Readiness: {ml_readiness:.1f}%
            - Missing Values: {missing_cells} ({(missing_cells/total_cells)*100:.1f}%)
            - Duplicates: {duplicate_count} ({(duplicate_count/len(df))*100:.1f}%)
            - Numeric Columns: {len(numeric_cols)}
            - Categorical Columns: {len(categorical_cols)}
            - High Correlations: {len(high_corr_pairs) if 'high_corr_pairs' in locals() else 0}
            - Normality Tests: {len([r for r in normality_results if r['Normal?'] == '✅ Yes']) if 'normality_results' in locals() else 'N/A'} normally distributed
            - Outliers Detected: Multiple methods applied
            """
            
            insights = generate_text("Provide comprehensive insights and actionable recommendations for this dataset analysis", analysis_context)
            ai_response(insights)
            
            # === INTERACTIVE DATA EXPLORER ===
            section_header("🔍", "13. Interactive Data Explorer")
            
            st.subheader("🎯 Custom Scatter Plot Analysis")
            
            if len(numeric_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_axis = st.selectbox("Select X-axis:", numeric_cols, key="scatter_x")
                with col2:
                    y_axis = st.selectbox("Select Y-axis:", [c for c in numeric_cols if c != x_axis], key="scatter_y")
                with col3:
                    color_by = st.selectbox("Color by (optional):", ["None"] + categorical_cols, key="scatter_color")
                
                if x_axis and y_axis:
                    if color_by != "None":
                        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, 
                                       title=f'{y_axis} vs {x_axis} (colored by {color_by})',
                                       trendline="ols")
                    else:
                        fig = px.scatter(df, x=x_axis, y=y_axis, 
                                       title=f'{y_axis} vs {x_axis}',
                                       trendline="ols")
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Calculate correlation
                    corr = df[[x_axis, y_axis]].corr().iloc[0, 1]
                    st.metric("Correlation Coefficient", f"{corr:.4f}")
            
            # === DATA QUALITY REPORT ===
            section_header("📋", "14. Comprehensive Data Quality Report")
            
            st.subheader("✅ Quality Checklist")
            
            quality_checks = []
            
            # Completeness check
            quality_checks.append({
                'Check': 'Data Completeness',
                'Status': '✅ Pass' if completeness_score >= 90 else '⚠️ Warning' if completeness_score >= 70 else '❌ Fail',
                'Score': f"{completeness_score:.1f}%",
                'Details': f"{missing_cells} missing values out of {total_cells} total cells"
            })
            
            # Uniqueness check
            quality_checks.append({
                'Check': 'Data Uniqueness',
                'Status': '✅ Pass' if duplicate_count == 0 else '⚠️ Warning' if duplicate_count < len(df)*0.05 else '❌ Fail',
                'Score': f"{uniqueness_score:.1f}%",
                'Details': f"{duplicate_count} duplicate rows found"
            })
            
            # Consistency check
            quality_checks.append({
                'Check': 'Data Consistency',
                'Status': '✅ Pass' if consistency_score >= 90 else '⚠️ Warning',
                'Score': f"{consistency_score:.1f}%",
                'Details': f"{consistency_issues} consistency issues in categorical data"
            })
            
            # Validity check (data types)
            type_issues = sum([1 for col in df.columns if df[col].dtype == 'object' and 
                             any(keyword in col.lower() for keyword in ['id', 'number', 'count', 'amount'])])
            
            quality_checks.append({
                'Check': 'Data Type Validity',
                'Status': '✅ Pass' if type_issues == 0 else '⚠️ Warning',
                'Score': f"{((len(df.columns)-type_issues)/len(df.columns)*100):.1f}%",
                'Details': f"{type_issues} potential data type mismatches"
            })
            
            # Outlier check
            total_outliers = sum([r['IQR Outliers'] for r in outlier_summary]) if 'outlier_summary' in locals() else 0
            outlier_ratio = (total_outliers / (len(df) * len(numeric_cols))) * 100 if numeric_cols else 0
            
            quality_checks.append({
                'Check': 'Outlier Assessment',
                'Status': '✅ Pass' if outlier_ratio < 5 else '⚠️ Warning' if outlier_ratio < 10 else '❌ Fail',
                'Score': f"{100-outlier_ratio:.1f}%",
                'Details': f"{total_outliers} outliers detected across numeric columns"
            })
            
            quality_df = pd.DataFrame(quality_checks)
            st.dataframe(quality_df, width='stretch', hide_index=True)
            
            # Overall quality grade
            pass_count = len([c for c in quality_checks if '✅' in c['Status']])
            total_checks = len(quality_checks)
            overall_grade = (pass_count / total_checks) * 100
            
            grade = "A" if overall_grade >= 90 else "B" if overall_grade >= 75 else "C" if overall_grade >= 60 else "D"
            grade_color = "🟢" if grade in ["A", "B"] else "🟡" if grade == "C" else "🔴"
            
            st.metric("Overall Data Quality Grade", f"{grade_color} Grade {grade} ({overall_grade:.0f}%)")
            
            # === EXPORT ANALYSIS SUMMARY ===
            section_header("💾", "15. Export Analysis Summary")
            
            st.subheader("📄 Generate Analysis Report")
            
            if st.button("📊 Generate Downloadable Report"):
                report_data = {
                    'Dataset Overview': {
                        'Rows': df.shape[0],
                        'Columns': df.shape[1],
                        'Memory Usage (MB)': f"{memory_usage:.2f}",
                        'Missing Values': missing_cells,
                        'Duplicates': duplicate_count
                    },
                    'Data Quality Scores': {
                        'Completeness': f"{completeness_score:.1f}%",
                        'Uniqueness': f"{uniqueness_score:.1f}%",
                        'Consistency': f"{consistency_score:.1f}%",
                        'Overall Quality': f"{overall_quality:.1f}%",
                        'ML Readiness': f"{ml_readiness:.1f}%"
                    },
                    'Column Information': {
                        'Numeric Columns': len(numeric_cols),
                        'Categorical Columns': len(categorical_cols),
                        'DateTime Columns': len(datetime_cols) if 'datetime_cols' in locals() else 0
                    }
                }
                
                report_df = pd.DataFrame(report_data)
                
                # Convert to CSV
                csv = report_df.to_csv(index=True)
                st.download_button(
                    label="📥 Download Analysis Report (CSV)",
                    data=csv,
                    file_name="data_analysis_report.csv",
                    mime="text/csv"
                )
                
                st.success("✅ Report generated successfully!")
            
            # Suggested next steps
            st.subheader("📋 Recommended Next Steps")
            
            recommendations = []
            
            if completeness_score < 90:
                recommendations.append("🔧 **Data Cleaning**: Address missing values using appropriate imputation strategies")
            
            if duplicate_count > 0:
                recommendations.append("🗑️ **Deduplication**: Remove or investigate duplicate records")
            
            if 'high_corr_pairs' in locals() and len(high_corr_pairs) > 0:
                recommendations.append("📊 **Feature Selection**: Address multicollinearity by removing highly correlated features")
            
            if ml_readiness < 80:
                recommendations.append("🎯 **Data Preparation**: Improve data quality before machine learning")
            
            if len(numeric_cols) > 10:
                recommendations.append("📐 **Dimensionality Reduction**: Consider PCA or feature selection techniques")
            
            if 'normality_results' in locals():
                non_normal = len([r for r in normality_results if r['Normal?'] == '❌ No'])
                if non_normal > 0:
                    recommendations.append(f"📈 **Data Transformation**: {non_normal} variables are non-normal, consider transformations")
            
            if 'outlier_summary' in locals() and total_outliers > 0:
                recommendations.append(f"🚨 **Outlier Treatment**: {total_outliers} outliers detected, review and handle appropriately")
            
            recommendations.append("🤖 **Model Training**: Use the Model Training section to build predictive models")
            recommendations.append("🔍 **Variable Analysis**: Dive deeper into individual variables for detailed insights")
            recommendations.append("❓ **Ask Questions**: Use Custom Question feature for specific data inquiries")
            
            for rec in recommendations:
                st.write(rec)
            
            # Final summary
            st.success(f"""
            ✅ **Analysis Complete!** 
            
            Analyzed {df.shape[0]:,} rows × {df.shape[1]} columns with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical variables.
            Data Quality Grade: {grade} | ML Readiness: {ml_readiness:.0f}%
            """)
            
            return
        
        def function_question_dataframe(user_question):
            # Provide comprehensive dataset context to Gemini
            dataset_context = f"""
            Dataset Information:
            - Shape: {df.shape[0]} rows, {df.shape[1]} columns
            - Columns: {list(df.columns)}
            - Data types: {dict(df.dtypes)}
            - Missing values per column: {dict(df.isnull().sum())}
            - Numeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}
            - Categorical columns: {df.select_dtypes(include=['object']).columns.tolist()}
            - Sample data (first 3 rows): {df.head(3).to_dict()}
            - Basic statistics for numeric columns: {df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else 'No numeric columns'}
            """
            
            # Enhanced prompt for better responses
            enhanced_question = f"""
            User Question: {user_question}
            
            Dataset Context: {dataset_context}
            
            Please provide a detailed, specific answer based on the actual data provided. 
            Include relevant statistics, insights, and actionable recommendations.
            """
            
            with st.spinner("Analyzing your data and generating response..."):
                dataframe_info = generate_text(enhanced_question, dataset_context)
                st.write("**AI Analysis Result:**")
                st.write(dataframe_info)
            return
            
            # === AI-POWERED INSIGHTS ===
            section_header("🤖", "5. AI-Powered Data Insights")
            
            # Prepare comprehensive context for AI
            dataset_summary = f"""
            Dataset Overview:
            - Shape: {df.shape}
            - Columns: {list(df.columns)}
            - Data types: {dict(df.dtypes)}
            - Missing values: {dict(df.isnull().sum())}
            - Numeric columns: {numeric_cols}
            - Categorical columns: {categorical_cols}
            - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
            - Duplicate rows: {duplicate_count}
            
            Statistical Summary:
            {df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else 'No numeric data'}
            """
            
            with st.spinner("🤖 Generating AI insights..."):
                # Overall insights
                overall_insights = generate_text(
                    "Provide comprehensive insights about this dataset including data quality, patterns, and recommendations for analysis",
                    dataset_summary
                )
                st.write("**📋 Overall Dataset Insights:**")
                st.write(overall_insights)
                
                # Data quality insights
                quality_insights = generate_text(
                    "Analyze the data quality issues and provide specific recommendations for data cleaning and preprocessing",
                    dataset_summary
                )
                st.write("**🔍 Data Quality Assessment:**")
                st.write(quality_insights)
                
                # ML recommendations
                ml_insights = generate_text(
                    "Based on this dataset structure and characteristics, recommend the most suitable machine learning approaches and models",
                    dataset_summary
                )
                st.write("**🤖 Machine Learning Recommendations:**")
                st.write(ml_insights)
            
            # === ACTIONABLE RECOMMENDATIONS ===
            section_header("💡", "6. Actionable Recommendations")
            
            recommendations = []
            
            # Data quality recommendations
            if missing_data.sum() > 0:
                recommendations.append("🔧 **Data Cleaning**: Handle missing values using appropriate imputation strategies")
            
            if duplicate_count > 0:
                recommendations.append("🔄 **Duplicate Removal**: Remove or investigate duplicate records")
            
            # Feature engineering recommendations
            if len(categorical_cols) > 0:
                recommendations.append("🏷️ **Feature Engineering**: Consider encoding categorical variables for ML models")
            
            if len(numeric_cols) > 1:
                recommendations.append("📊 **Feature Selection**: Analyze correlations to identify redundant features")
            
            # Analysis recommendations
            recommendations.append("🔍 **Deep Dive**: Use Variable Analysis for detailed column-specific insights")
            recommendations.append("🤖 **Model Training**: Try the Model Training section with different algorithms")
            recommendations.append("❓ **Custom Questions**: Ask specific questions about your data using the Custom Question feature")
            
            for rec in recommendations:
                st.write(rec)
            
            st.success("✅ Comprehensive analysis completed! Use the sidebar to explore specific features.")
        
        def function_question_variable(selected_variable):
            # Get variable statistics for context
            var_stats = df[selected_variable].describe() if df[selected_variable].dtype in ['float64', 'int64'] else df[selected_variable].value_counts()
            missing_count = df[selected_variable].isnull().sum()
            data_type = str(df[selected_variable].dtype)
            is_numeric = df[selected_variable].dtype in ['float64', 'int64']

            context = f"Variable: {selected_variable}, Data type: {data_type}, Missing values: {missing_count}, Statistics: {var_stats}"

            # ── Plain-English header ──
            st.markdown(f"""
            <div style='background:#f0f4ff; border-left:5px solid #667eea; border-radius:0 12px 12px 0;
                        padding:18px 20px; margin-bottom:20px;'>
                <div style='font-size:1.4em; font-weight:700; color:#1e1b4b;'>🔍 Analysing: <span style='color:#667eea;'>{selected_variable}</span></div>
                <div style='color:#555; margin-top:6px; font-size:0.95em;'>
                    This section helps you understand everything about this one column in your dataset —
                    what kind of data it holds, how spread out the values are, whether anything looks unusual,
                    and what it means in plain language.
                </div>
            </div>""", unsafe_allow_html=True)

            # ── What is this column? ──
            type_plain = "Numbers (measurements, counts, prices, etc.)" if is_numeric else "Categories / Text (labels, names, groups, etc.)"
            st.markdown(f"""
            <div style='background:#fff; border:1px solid #e0e7ff; border-radius:10px; padding:15px 18px; margin-bottom:15px;'>
                <b>📌 What kind of data is this?</b><br>
                <span style='color:#5b21b6;'>{type_plain}</span><br>
                <span style='color:#666; font-size:0.88em;'>Technical type: <code>{data_type}</code></span>
            </div>""", unsafe_allow_html=True)

            # ── Key metrics ──
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Type", "Numeric" if is_numeric else "Categorical")
            with col2:
                st.metric("Missing Values", f"{missing_count} ({missing_count/len(df)*100:.1f}%)")
            with col3:
                st.metric("Unique Values" if is_numeric else "Categories", df[selected_variable].nunique())

            # ── Plain-English stats for numeric ──
            if is_numeric:
                col_data = df[selected_variable].dropna()
                mean_val  = col_data.mean()
                median_val= col_data.median()
                std_val   = col_data.std()
                min_val   = col_data.min()
                max_val   = col_data.max()
                skew_val  = col_data.skew()

                skew_plain = ("fairly balanced (close to normal)" if abs(skew_val) < 0.5
                              else "slightly skewed — more values on one side" if abs(skew_val) < 1
                              else "heavily skewed — a few very large or very small values pull the average")

                st.markdown(f"""
                <div style='background:#f9fafb; border:1px solid #e0e7ff; border-radius:12px; padding:18px; margin:15px 0;'>
                    <b style='color:#1e1b4b; font-size:1.05em;'>📊 What the numbers tell us (plain English)</b>
                    <ul style='margin-top:10px; color:#333; line-height:1.9;'>
                        <li><b>Average (Mean):</b> {mean_val:,.2f} — this is the typical value you'd expect</li>
                        <li><b>Middle value (Median):</b> {median_val:,.2f} — half the data is above this, half below</li>
                        <li><b>Spread (Std Dev):</b> {std_val:,.2f} — values typically vary by this much from the average</li>
                        <li><b>Range:</b> {min_val:,.2f} → {max_val:,.2f} — the lowest and highest values in the data</li>
                        <li><b>Shape:</b> The distribution is <em>{skew_plain}</em></li>
                        {"<li><b>⚠️ Note:</b> Mean and median are quite different — this suggests outliers or skewed data</li>" if abs(mean_val - median_val) > std_val * 0.5 else ""}
                    </ul>
                </div>""", unsafe_allow_html=True)

                # ── Charts ──
                st.markdown("**📈 Distribution & Spread Charts**")
                st.caption("The histogram shows how often each value appears. The box plot shows the spread and any unusual values (dots outside the box are outliers).")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
                df[selected_variable].hist(bins=30, ax=ax1, color='#667eea', edgecolor='white')
                ax1.set_title(f'How often each value appears\n({selected_variable})')
                ax1.set_xlabel(selected_variable)
                ax1.set_ylabel('Count')
                df.boxplot(column=selected_variable, ax=ax2, patch_artist=True,
                           boxprops=dict(facecolor='#e0e7ff', color='#667eea'),
                           medianprops=dict(color='#764ba2', linewidth=2))
                ax2.set_title(f'Spread & outliers\n({selected_variable})')
                st.pyplot(fig)

                # Outlier plain explanation
                Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = len(col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)])
                if outlier_count > 0:
                    st.markdown(f"""
                    <div style='background:#fff7ed; border-left:4px solid #f59e0b; border-radius:0 8px 8px 0; padding:12px 16px; margin:10px 0;'>
                        ⚠️ <b>{outlier_count} unusual values (outliers)</b> were found — these are values that are much higher or lower than the rest.
                        They could be data entry errors, or genuinely rare events worth investigating.
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div style='background:#f0fdf4; border-left:4px solid #10b981; border-radius:0 8px 8px 0; padding:12px 16px; margin:10px 0;'>
                        ✅ <b>No unusual outliers found</b> — the data looks clean and consistent.</div>""", unsafe_allow_html=True)

                st.markdown("**📉 Trend over rows**")
                st.caption("This shows how the value changes row by row — useful to spot trends or patterns over time.")
                st.line_chart(df[selected_variable])

            else:
                # ── Categorical ──
                value_counts = df[selected_variable].value_counts()
                top_cat = value_counts.index[0]
                top_pct = value_counts.iloc[0] / len(df) * 100
                n_cats  = df[selected_variable].nunique()

                st.markdown(f"""
                <div style='background:#f9fafb; border:1px solid #e0e7ff; border-radius:12px; padding:18px; margin:15px 0;'>
                    <b style='color:#1e1b4b; font-size:1.05em;'>📊 What the categories tell us (plain English)</b>
                    <ul style='margin-top:10px; color:#333; line-height:1.9;'>
                        <li><b>Number of categories:</b> {n_cats} different values</li>
                        <li><b>Most common:</b> "{top_cat}" — appears in {top_pct:.1f}% of rows</li>
                        <li><b>Least common:</b> "{value_counts.index[-1]}" — appears only {value_counts.iloc[-1]} times</li>
                        {"<li><b>⚠️ Imbalanced:</b> One category dominates — this can affect ML model fairness</li>" if top_pct > 70 else "<li><b>✅ Reasonably balanced</b> across categories</li>"}
                    </ul>
                </div>""", unsafe_allow_html=True)

                st.markdown("**📊 Category Frequency Chart**")
                st.caption("Each bar shows how many rows belong to that category. Taller bar = more common.")
                st.bar_chart(value_counts.head(15))
                st.write("**Top 10 categories:**")
                st.dataframe(value_counts.head(10).reset_index().rename(columns={'index': 'Category', selected_variable: 'Count'}))

            # ── Missing values plain explanation ──
            if missing_count > 0:
                pct = missing_count / len(df) * 100
                severity = "🔴 High — this may affect analysis quality" if pct > 30 else "🟡 Moderate — consider filling or removing" if pct > 10 else "🟢 Low — minor impact"
                st.markdown(f"""
                <div style='background:#fff7ed; border-left:4px solid #f59e0b; border-radius:0 8px 8px 0; padding:12px 16px; margin:10px 0;'>
                    <b>Missing Data:</b> {missing_count} rows ({pct:.1f}%) have no value here.<br>
                    Severity: {severity}<br>
                    <span style='font-size:0.88em; color:#666;'>💡 Tip: You can fill missing numbers with the average, or fill missing categories with "Unknown".</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style='background:#f0fdf4; border-left:4px solid #10b981; border-radius:0 8px 8px 0; padding:12px 16px; margin:10px 0;'>
                    ✅ <b>No missing values</b> — this column is complete.</div>""", unsafe_allow_html=True)

            # ── AI Insights ──
            st.markdown("---")
            st.markdown("**🤖 AI Insights**")
            with st.spinner("Generating AI analysis..."):
                summary_statistics = generate_text(
                    f"Explain in simple non-technical language what the variable '{selected_variable}' represents and what its statistics mean. Data: {context}",
                    context
                )
            ai_response(summary_statistics)

            if is_numeric:
                with st.spinner("Analysing distribution..."):
                    normality = generate_text(
                        f"In plain English for a non-technical audience, explain the distribution shape of '{selected_variable}' and what it means practically. Data: {context}",
                        context
                    )
                ai_response(normality)
            else:
                with st.spinner("Analysing categories..."):
                    category_analysis = generate_text(
                        f"In plain English, explain the category distribution of '{selected_variable}' and what patterns or concerns exist. Data: {context}",
                        context
                    )
                ai_response(category_analysis)

            return
        
        def function_question_dataframe(user_question):
            # Provide comprehensive dataset context to Gemini
            dataset_context = f"""
            Dataset Information:
            - Shape: {df.shape[0]} rows, {df.shape[1]} columns
            - Columns: {list(df.columns)}
            - Data types: {dict(df.dtypes)}
            - Missing values per column: {dict(df.isnull().sum())}
            - Numeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}
            - Categorical columns: {df.select_dtypes(include=['object']).columns.tolist()}
            - Sample data (first 3 rows): {df.head(3).to_dict()}
            - Basic statistics for numeric columns: {df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else 'No numeric columns'}
            """
            enhanced_question = f"""
            User Question: {user_question}
            Dataset Context: {dataset_context}
            Please provide a detailed, specific answer based on the actual data provided. 
            Include relevant statistics, insights, and actionable recommendations.
            """
            dataframe_info = generate_text(enhanced_question, dataset_context)
            return dataframe_info
        
        def train_model_enhanced(target_col, feature_cols, selected_models, problem_type, test_size, random_state, use_cv, cv_folds):
            from sklearn.linear_model import Ridge, Lasso, LogisticRegression
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error, confusion_matrix
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder
            import time
            import pickle
            import plotly.figure_factory as ff
            
            # Validate inputs
            if target_col not in df.columns:
                st.error(f"Target column '{target_col}' does not exist in the dataset.")
                return
            if not set(feature_cols).issubset(df.columns):
                st.error("One or more feature columns do not exist in the dataset.")
                return
            
            # Prepare the data
            X = df[feature_cols].select_dtypes(include=[np.number])
            y = df[target_col]
            
            if X.empty:
                st.error("No numeric features selected for training.")
                return
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Handle target variable based on problem type
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
            
            # Define model dictionaries
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
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
                'Support Vector Machine': SVC(random_state=random_state),
                'Decision Tree': DecisionTreeClassifier(random_state=random_state)
            }
            
            models_dict = regression_models if problem_type == "Regression" else classification_models
            
            # Filter selected models
            models_to_train = {name: model for name, model in models_dict.items() if name in selected_models}
            
            st.write(f"**Training {len(models_to_train)} models on {X.shape[0]} samples with {X.shape[1]} features...**")
            
            results = {}
            training_times = {}
            trained_models = {}  # Store trained models for download
            predictions = {}  # Store predictions for confusion matrix
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (name, model) in enumerate(models_to_train.items()):
                status_text.text(f"Training {name}...")
                
                try:
                    start_time = time.time()
                    
                    # Train model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    training_time = time.time() - start_time
                    training_times[name] = training_time
                    
                    # Store trained model and predictions
                    trained_models[name] = model
                    predictions[name] = {'y_test': y_test, 'y_pred': y_pred}
                    
                    # Calculate metrics based on problem type
                    if problem_type == "Regression":
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        results[name] = {
                            'MSE': mse,
                            'MAE': mae,
                            'R²': r2,
                            'RMSE': np.sqrt(mse)
                        }
                        
                        # Cross-validation if enabled
                        if use_cv:
                            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                            results[name]['CV_R²_Mean'] = cv_scores.mean()
                            results[name]['CV_R²_Std'] = cv_scores.std()
                    
                    else:  # Classification
                        accuracy = accuracy_score(y_test, y_pred)
                        results[name] = {'Accuracy': accuracy}
                        
                        # Cross-validation if enabled
                        if use_cv:
                            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
                            results[name]['CV_Accuracy_Mean'] = cv_scores.mean()
                            results[name]['CV_Accuracy_Std'] = cv_scores.std()
                
                except Exception as e:
                    st.error(f"{name} - Training Error: {str(e)}")
                    results[name] = {'Error': str(e)}
                
                # Update progress
                progress_bar.progress((i + 1) / len(models_to_train))
            
            status_text.text("Training completed!")
            
            # Display results
            if results:
                section_header("📊", "Model Performance Results")

                # Plain-English metric explainer
                if problem_type == "Regression":
                    st.markdown("""
                    <div style='background:#f0f4ff; border:1px solid #c7d2fe; border-radius:10px; padding:14px 18px; margin-bottom:15px;'>
                        <b>📖 How to read these results:</b>
                        <ul style='margin-top:8px; color:#333; line-height:1.8;'>
                            <li><b>R² (R-squared):</b> How well the model explains the data. <b>1.0 = perfect</b>, 0 = no better than guessing the average. Above 0.8 is generally good.</li>
                            <li><b>MAE (Mean Absolute Error):</b> On average, how far off are the predictions? Lower is better. Same units as your target column.</li>
                            <li><b>RMSE:</b> Similar to MAE but penalises big errors more. Lower is better.</li>
                            <li><b>CV R²:</b> R² tested across multiple data splits — more reliable than a single test.</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background:#f0f4ff; border:1px solid #c7d2fe; border-radius:10px; padding:14px 18px; margin-bottom:15px;'>
                        <b>📖 How to read these results:</b>
                        <ul style='margin-top:8px; color:#333; line-height:1.8;'>
                            <li><b>Accuracy:</b> What % of predictions were correct. <b>1.0 = 100% correct</b>. Above 0.85 is generally good.</li>
                            <li><b>CV Accuracy:</b> Accuracy tested across multiple data splits — more reliable than a single test.</li>
                            <li><b>CV Std:</b> How consistent the model is. Lower = more stable and trustworthy.</li>
                        </ul>
                    </div>""", unsafe_allow_html=True)

                # Create results DataFrame
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df.round(4), width='stretch')

                # Find best model
                if problem_type == "Regression":
                    valid_results = {k: v for k, v in results.items() if 'R²' in v}
                    if valid_results:
                        best_model = max(valid_results.keys(), key=lambda x: valid_results[x]['R²'])
                        r2 = valid_results[best_model]['R²']
                        quality = "Excellent" if r2 > 0.9 else "Good" if r2 > 0.75 else "Fair" if r2 > 0.5 else "Poor"
                        st.success(f"🏆 Best Model: **{best_model}** — R² = {r2:.4f} ({quality})")
                        st.info(f"💡 In plain terms: **{best_model}** can explain **{r2*100:.1f}%** of the variation in your target variable.")
                else:
                    valid_results = {k: v for k, v in results.items() if 'Accuracy' in v}
                    if valid_results:
                        best_model = max(valid_results.keys(), key=lambda x: valid_results[x]['Accuracy'])
                        acc = valid_results[best_model]['Accuracy']
                        quality = "Excellent" if acc > 0.9 else "Good" if acc > 0.8 else "Fair" if acc > 0.65 else "Needs improvement"
                        st.success(f"🏆 Best Model: **{best_model}** — Accuracy = {acc:.4f} ({quality})")
                        st.info(f"💡 In plain terms: **{best_model}** correctly predicts **{acc*100:.1f}%** of cases.")
                
                # Training time comparison
                st.write("## ⏱️ Training Time Comparison")
                time_df = pd.DataFrame(list(training_times.items()), columns=['Model', 'Training Time (s)'])
                st.bar_chart(time_df.set_index('Model'))
                
                # Confusion Matrix for Classification
                if problem_type == "Classification" and predictions:
                    st.write("## 📊 Confusion Matrix")
                    
                    # Let user select which model to view
                    model_for_cm = st.selectbox(
                        "Select model for confusion matrix:",
                        list(predictions.keys()),
                        key="cm_model_select"
                    )
                    
                    if model_for_cm:
                        y_test_cm = predictions[model_for_cm]['y_test']
                        y_pred_cm = predictions[model_for_cm]['y_pred']
                        
                        # Calculate confusion matrix
                        cm = confusion_matrix(y_test_cm, y_pred_cm)
                        
                        # Get unique labels
                        labels = sorted(list(set(y_test_cm) | set(y_pred_cm)))
                        
                        # Create confusion matrix visualization
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Plotly heatmap
                            fig = px.imshow(cm, 
                                          labels=dict(x="Predicted", y="Actual", color="Count"),
                                          x=[f"Class {i}" for i in labels],
                                          y=[f"Class {i}" for i in labels],
                                          title=f'Confusion Matrix - {model_for_cm}',
                                          color_continuous_scale='Blues',
                                          text_auto=True)
                            fig.update_layout(width=600, height=500)
                            st.plotly_chart(fig, width='stretch')
                        
                        with col2:
                            st.write("**Confusion Matrix Metrics:**")
                            
                            # Calculate metrics from confusion matrix
                            if len(labels) == 2:  # Binary classification
                                tn, fp, fn, tp = cm.ravel()
                                
                                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                                
                                st.metric("Precision", f"{precision:.4f}")
                                st.metric("Recall", f"{recall:.4f}")
                                st.metric("F1-Score", f"{f1:.4f}")
                                st.metric("True Positives", tp)
                                st.metric("True Negatives", tn)
                                st.metric("False Positives", fp)
                                st.metric("False Negatives", fn)
                            else:  # Multi-class
                                accuracy = np.trace(cm) / np.sum(cm)
                                st.metric("Overall Accuracy", f"{accuracy:.4f}")
                                st.metric("Total Predictions", int(np.sum(cm)))
                                st.metric("Correct Predictions", int(np.trace(cm)))
                        
                        # Classification Report
                        st.write("**📋 Classification Report:**")
                        report = classification_report(y_test_cm, y_pred_cm, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.round(4), width='stretch')
                
                # Comprehensive Evaluation Metrics
                st.write("## 📈 Comprehensive Evaluation Metrics")
                
                if problem_type == "Classification":
                    st.subheader("🎯 Classification Evaluation Metrics")
                    
                    # Select model for detailed evaluation
                    model_for_eval = st.selectbox(
                        "Select model for detailed evaluation:",
                        list(predictions.keys()),
                        key="eval_model_select"
                    )
                    
                    if model_for_eval:
                        from sklearn.metrics import (
                            precision_score, recall_score, f1_score, 
                            roc_auc_score, log_loss, matthews_corrcoef,
                            cohen_kappa_score, balanced_accuracy_score
                        )
                        
                        y_test_eval = predictions[model_for_eval]['y_test']
                        y_pred_eval = predictions[model_for_eval]['y_pred']
                        
                        # Get probability predictions if available
                        model_obj = trained_models[model_for_eval]
                        y_pred_proba = None
                        if hasattr(model_obj, 'predict_proba'):
                            try:
                                y_pred_proba = model_obj.predict_proba(X_test)
                            except:
                                pass
                        
                        # Calculate comprehensive metrics
                        eval_metrics = {}
                        
                        # Basic metrics
                        eval_metrics['Accuracy'] = accuracy_score(y_test_eval, y_pred_eval)
                        eval_metrics['Balanced Accuracy'] = balanced_accuracy_score(y_test_eval, y_pred_eval)
                        
                        # Determine if binary or multi-class
                        n_classes = len(np.unique(y_test_eval))
                        
                        if n_classes == 2:
                            # Binary classification metrics
                            eval_metrics['Precision'] = precision_score(y_test_eval, y_pred_eval, zero_division=0)
                            eval_metrics['Recall'] = recall_score(y_test_eval, y_pred_eval, zero_division=0)
                            eval_metrics['F1-Score'] = f1_score(y_test_eval, y_pred_eval, zero_division=0)
                            eval_metrics['Matthews Correlation'] = matthews_corrcoef(y_test_eval, y_pred_eval)
                            eval_metrics['Cohen Kappa'] = cohen_kappa_score(y_test_eval, y_pred_eval)
                            
                            # ROC-AUC if probabilities available
                            if y_pred_proba is not None:
                                eval_metrics['ROC-AUC'] = roc_auc_score(y_test_eval, y_pred_proba[:, 1])
                                eval_metrics['Log Loss'] = log_loss(y_test_eval, y_pred_proba)
                        else:
                            # Multi-class metrics
                            eval_metrics['Precision (Macro)'] = precision_score(y_test_eval, y_pred_eval, average='macro', zero_division=0)
                            eval_metrics['Precision (Weighted)'] = precision_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
                            eval_metrics['Recall (Macro)'] = recall_score(y_test_eval, y_pred_eval, average='macro', zero_division=0)
                            eval_metrics['Recall (Weighted)'] = recall_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
                            eval_metrics['F1-Score (Macro)'] = f1_score(y_test_eval, y_pred_eval, average='macro', zero_division=0)
                            eval_metrics['F1-Score (Weighted)'] = f1_score(y_test_eval, y_pred_eval, average='weighted', zero_division=0)
                            eval_metrics['Cohen Kappa'] = cohen_kappa_score(y_test_eval, y_pred_eval)
                            
                            # Multi-class ROC-AUC if probabilities available
                            if y_pred_proba is not None:
                                try:
                                    eval_metrics['ROC-AUC (OvR)'] = roc_auc_score(y_test_eval, y_pred_proba, multi_class='ovr')
                                    eval_metrics['Log Loss'] = log_loss(y_test_eval, y_pred_proba)
                                except:
                                    pass
                        
                        # Display metrics in columns
                        col1, col2, col3 = st.columns(3)
                        
                        metrics_list = list(eval_metrics.items())
                        third = len(metrics_list) // 3 + 1
                        
                        with col1:
                            for metric, value in metrics_list[:third]:
                                st.metric(metric, f"{value:.4f}")
                        
                        with col2:
                            for metric, value in metrics_list[third:2*third]:
                                st.metric(metric, f"{value:.4f}")
                        
                        with col3:
                            for metric, value in metrics_list[2*third:]:
                                st.metric(metric, f"{value:.4f}")
                        
                        # ROC Curve for binary classification
                        if n_classes == 2 and y_pred_proba is not None:
                            from sklearn.metrics import roc_curve, auc
                            
                            st.write("**📊 ROC Curve:**")
                            
                            fpr, tpr, thresholds = roc_curve(y_test_eval, y_pred_proba[:, 1])
                            roc_auc = auc(fpr, tpr)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                                   name=f'ROC Curve (AUC = {roc_auc:.4f})',
                                                   line=dict(color='blue', width=2)))
                            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                                   name='Random Classifier',
                                                   line=dict(color='red', width=2, dash='dash')))
                            fig.update_layout(
                                title=f'ROC Curve - {model_for_eval}',
                                xaxis_title='False Positive Rate',
                                yaxis_title='True Positive Rate',
                                width=700, height=500
                            )
                            st.plotly_chart(fig, width='stretch')
                        
                        # Precision-Recall Curve for binary classification
                        if n_classes == 2 and y_pred_proba is not None:
                            from sklearn.metrics import precision_recall_curve, average_precision_score
                            
                            st.write("**📊 Precision-Recall Curve:**")
                            
                            precision_curve, recall_curve, _ = precision_recall_curve(y_test_eval, y_pred_proba[:, 1])
                            avg_precision = average_precision_score(y_test_eval, y_pred_proba[:, 1])
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=recall_curve, y=precision_curve, mode='lines',
                                                   name=f'PR Curve (AP = {avg_precision:.4f})',
                                                   line=dict(color='green', width=2)))
                            fig.update_layout(
                                title=f'Precision-Recall Curve - {model_for_eval}',
                                xaxis_title='Recall',
                                yaxis_title='Precision',
                                width=700, height=500
                            )
                            st.plotly_chart(fig, width='stretch')
                
                else:  # Regression
                    st.subheader("📊 Regression Evaluation Metrics")
                    
                    # Select model for detailed evaluation
                    model_for_eval = st.selectbox(
                        "Select model for detailed evaluation:",
                        list(predictions.keys()),
                        key="eval_model_select_reg"
                    )
                    
                    if model_for_eval:
                        from sklearn.metrics import (
                            mean_absolute_percentage_error, median_absolute_error,
                            max_error, explained_variance_score
                        )
                        
                        y_test_eval = predictions[model_for_eval]['y_test']
                        y_pred_eval = predictions[model_for_eval]['y_pred']
                        
                        # Calculate comprehensive regression metrics
                        eval_metrics = {}
                        
                        eval_metrics['R² Score'] = r2_score(y_test_eval, y_pred_eval)
                        eval_metrics['Adjusted R²'] = 1 - (1 - r2_score(y_test_eval, y_pred_eval)) * (len(y_test_eval) - 1) / (len(y_test_eval) - X_test.shape[1] - 1)
                        eval_metrics['Mean Squared Error (MSE)'] = mean_squared_error(y_test_eval, y_pred_eval)
                        eval_metrics['Root Mean Squared Error (RMSE)'] = np.sqrt(mean_squared_error(y_test_eval, y_pred_eval))
                        eval_metrics['Mean Absolute Error (MAE)'] = mean_absolute_error(y_test_eval, y_pred_eval)
                        eval_metrics['Median Absolute Error'] = median_absolute_error(y_test_eval, y_pred_eval)
                        eval_metrics['Mean Absolute Percentage Error (MAPE)'] = mean_absolute_percentage_error(y_test_eval, y_pred_eval) * 100
                        eval_metrics['Max Error'] = max_error(y_test_eval, y_pred_eval)
                        eval_metrics['Explained Variance'] = explained_variance_score(y_test_eval, y_pred_eval)
                        
                        # Display metrics in columns
                        col1, col2, col3 = st.columns(3)
                        
                        metrics_list = list(eval_metrics.items())
                        third = len(metrics_list) // 3 + 1
                        
                        with col1:
                            for metric, value in metrics_list[:third]:
                                if 'Percentage' in metric:
                                    st.metric(metric, f"{value:.2f}%")
                                else:
                                    st.metric(metric, f"{value:.4f}")
                        
                        with col2:
                            for metric, value in metrics_list[third:2*third]:
                                if 'Percentage' in metric:
                                    st.metric(metric, f"{value:.2f}%")
                                else:
                                    st.metric(metric, f"{value:.4f}")
                        
                        with col3:
                            for metric, value in metrics_list[2*third:]:
                                if 'Percentage' in metric:
                                    st.metric(metric, f"{value:.2f}%")
                                else:
                                    st.metric(metric, f"{value:.4f}")
                        
                        # Residual Plot
                        st.write("**📊 Residual Plot:**")
                        
                        residuals = y_test_eval - y_pred_eval
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=y_pred_eval, y=residuals, mode='markers',
                                               marker=dict(color='blue', size=8, opacity=0.6),
                                               name='Residuals'))
                        fig.add_trace(go.Scatter(x=[y_pred_eval.min(), y_pred_eval.max()], y=[0, 0],
                                               mode='lines', line=dict(color='red', width=2, dash='dash'),
                                               name='Zero Line'))
                        fig.update_layout(
                            title=f'Residual Plot - {model_for_eval}',
                            xaxis_title='Predicted Values',
                            yaxis_title='Residuals',
                            width=700, height=500
                        )
                        st.plotly_chart(fig, width='stretch')
                        
                        # Actual vs Predicted Plot
                        st.write("**📊 Actual vs Predicted:**")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=y_test_eval, y=y_pred_eval, mode='markers',
                                               marker=dict(color='blue', size=8, opacity=0.6),
                                               name='Predictions'))
                        fig.add_trace(go.Scatter(x=[y_test_eval.min(), y_test_eval.max()], 
                                               y=[y_test_eval.min(), y_test_eval.max()],
                                               mode='lines', line=dict(color='red', width=2, dash='dash'),
                                               name='Perfect Prediction'))
                        fig.update_layout(
                            title=f'Actual vs Predicted - {model_for_eval}',
                            xaxis_title='Actual Values',
                            yaxis_title='Predicted Values',
                            width=700, height=500
                        )
                        st.plotly_chart(fig, width='stretch')
                        
                        # Error Distribution
                        st.write("**📊 Error Distribution:**")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=residuals, nbinsx=30,
                                                 marker=dict(color='skyblue', line=dict(color='black', width=1)),
                                                 name='Residuals'))
                        fig.update_layout(
                            title=f'Residual Distribution - {model_for_eval}',
                            xaxis_title='Residual Value',
                            yaxis_title='Frequency',
                            width=700, height=400
                        )
                        st.plotly_chart(fig, width='stretch')
                
                # Model Download Section
                st.write("## 💾 Download Trained Models")
                
                if trained_models:
                    st.write("Select a model to download:")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        model_to_download = st.selectbox(
                            "Choose model:",
                            list(trained_models.keys()),
                            key="download_model_select"
                        )
                    
                    with col2:
                        if model_to_download:
                            # Serialize model
                            model_bytes = pickle.dumps(trained_models[model_to_download])
                            
                            # Create filename
                            filename = f"trained_model_{model_to_download.replace(' ', '_')}.pkl"
                            
                            st.download_button(
                                label=f"📥 Download {model_to_download}",
                                data=model_bytes,
                                file_name=filename,
                                mime="application/octet-stream",
                                help=f"Download the trained {model_to_download} model"
                            )
                    
                    # Model info
                    st.info(f"""
                    **Model Information:**
                    - Model: {model_to_download}
                    - Features: {len(feature_cols)} ({', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''})
                    - Target: {target_col}
                    - Problem Type: {problem_type}
                    - Training Samples: {len(X_train)}
                    - Test Samples: {len(X_test)}
                    
                    **How to use the downloaded model:**
                    ```python
                    import pickle
                    
                    # Load the model
                    with open('{filename}', 'rb') as f:
                        model = pickle.load(f)
                    
                    # Make predictions
                    predictions = model.predict(X_new)
                    ```
                    """)
                
                # Model insights
                insights_prompt = f"""
                Analyze these machine learning results for a {problem_type.lower()} problem:
                
                Models trained: {list(results.keys())}
                Best performing model: {best_model if 'best_model' in locals() else 'N/A'}
                Dataset size: {X.shape[0]} samples, {X.shape[1]} features
                
                Provide insights about:
                1. Why the best model performed well
                2. Recommendations for improvement
                3. Which model to use in production
                """
                
                ai_insights = generate_text(insights_prompt)
                st.write("## 🤖 AI Insights")
                st.write(ai_insights)
        
        if analysis_type == "Full Data Analysis":
            function_agent()
        
        elif analysis_type == "Variable Analysis":
            user_question_variable = st.selectbox('What variable are you interested in?', df.columns)
            if user_question_variable:
                if st.button("Analyze Variable", key=f"analyze_{user_question_variable}"):
                    # Clear any previous analysis
                    st.empty()
                    function_question_variable(user_question_variable)
        
        elif analysis_type == "Custom Question":
            section_header("💬", "AI Chat — Ask Anything About Your Data")
            
            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Render chat history
            if st.session_state.chat_history:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-msg-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-msg-ai"><div class="ai-label">🤖 AI Assistant</div>{msg["content"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align:center; padding:30px; color:#aaa;'>
                    <div style='font-size:3em;'>💬</div>
                    <div style='font-size:1.1em; margin-top:10px;'>Ask me anything about your dataset</div>
                    <div style='font-size:0.85em; margin-top:5px; color:#667eea;'>Powered by Google Gemini AI</div>
                </div>""", unsafe_allow_html=True)
            
            # Quick question chips
            st.markdown("**Quick Questions:**")
            example_questions = [
                "What are the key insights from this dataset?",
                "Which variables are most correlated?",
                "What data quality issues should I be aware of?",
                "What ML approach would work best?",
                "Are there any obvious patterns or trends?"
            ]
            cols = st.columns(len(example_questions))
            for i, (col, q) in enumerate(zip(cols, example_questions)):
                with col:
                    if st.button(q[:30]+"…", key=f"chip_{i}", help=q):
                        with st.spinner("🤖 Thinking..."):
                            answer = function_question_dataframe(q)
                        st.session_state.chat_history.append({"role": "user", "content": q})
                        st.session_state.chat_history.append({"role": "ai", "content": answer or "Analysis complete."})
                        st.rerun()
            
            # Text input
            col_input, col_btn = st.columns([5, 1])
            with col_input:
                user_question_dataframe = st.text_input(
                    "Your question", label_visibility="collapsed",
                    placeholder="e.g., What are the main patterns in this data?",
                    key="chat_input"
                )
            with col_btn:
                send = st.button("Send 🚀", use_container_width=True)
            
            if send and user_question_dataframe:
                with st.spinner("🤖 Generating AI insights..."):
                    answer = function_question_dataframe(user_question_dataframe)
                st.session_state.chat_history.append({"role": "user", "content": user_question_dataframe})
                st.session_state.chat_history.append({"role": "ai", "content": answer or "Analysis complete."})
                st.rerun()
            
            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        elif analysis_type == "Model Training":
            section_header("🤖", "Machine Learning Model Training")

            # ── Plain-English intro ──
            st.markdown("""
            <div style='background:#f0f4ff; border-left:5px solid #667eea; border-radius:0 12px 12px 0; padding:18px 20px; margin-bottom:20px;'>
                <b style='font-size:1.1em; color:#1e1b4b;'>🧠 What is Machine Learning?</b>
                <p style='color:#444; margin-top:8px; line-height:1.7;'>
                    Machine learning lets a computer <b>learn patterns from your data</b> and then make predictions on new data —
                    without being explicitly programmed. Think of it like teaching a student with examples, then testing them on new questions.
                </p>
                <p style='color:#444; margin-top:4px; line-height:1.7;'>
                    You pick a <b>target</b> (what you want to predict) and <b>features</b> (the information used to make that prediction).
                    The app will train multiple models and tell you which one performs best.
                </p>
            </div>""", unsafe_allow_html=True)

            # Check if we have enough numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            if len(numeric_cols) < 2:
                st.error("Need at least 2 numeric columns for model training.")
                st.info("💡 Tip: Try converting categorical columns to numeric or use feature encoding.")
            else:
                # ── Step 1 ──
                st.markdown("""
                <div style='background:#fff; border:1px solid #e0e7ff; border-radius:10px; padding:15px 18px; margin:10px 0;'>
                    <b style='color:#5b21b6;'>Step 1 — What kind of prediction do you want to make?</b><br>
                    <span style='color:#555; font-size:0.92em;'>
                    • <b>Regression</b> = predict a <em>number</em> (e.g. "What will the sales be next month?")<br>
                    • <b>Classification</b> = predict a <em>category</em> (e.g. "Is this email spam or not?")
                    </span>
                </div>""", unsafe_allow_html=True)
                problem_type = st.radio(
                    "Problem Type",
                    ["Regression", "Classification"],
                    label_visibility="collapsed",
                    help="Regression: predict numbers. Classification: predict categories."
                )

                # ── Step 2 ──
                st.markdown("""
                <div style='background:#fff; border:1px solid #e0e7ff; border-radius:10px; padding:15px 18px; margin:10px 0;'>
                    <b style='color:#5b21b6;'>Step 2 — What do you want to predict? (Target)</b><br>
                    <span style='color:#555; font-size:0.92em;'>
                    This is the column your model will try to predict. For example, if you want to predict house prices, select the "Price" column.
                    </span>
                </div>""", unsafe_allow_html=True)
                if problem_type == "Classification":
                    all_cols = numeric_cols + categorical_cols
                    target_col = st.selectbox("Target column (what to predict)", all_cols)
                else:
                    target_col = st.selectbox("Target column (what to predict)", numeric_cols)

                # ── Step 3 ──
                st.markdown("""
                <div style='background:#fff; border:1px solid #e0e7ff; border-radius:10px; padding:15px 18px; margin:10px 0;'>
                    <b style='color:#5b21b6;'>Step 3 — What information will the model use? (Features)</b><br>
                    <span style='color:#555; font-size:0.92em;'>
                    Features are the columns the model uses as clues to make its prediction.
                    More relevant features = better predictions. Avoid columns that directly reveal the answer.
                    </span>
                </div>""", unsafe_allow_html=True)
                available_features = [col for col in numeric_cols if col != target_col]
                feature_cols = st.multiselect(
                    "Feature columns (inputs for prediction)",
                    available_features,
                    default=available_features[:5] if len(available_features) >= 5 else available_features
                )

                # ── Step 4 ──
                st.markdown("""
                <div style='background:#fff; border:1px solid #e0e7ff; border-radius:10px; padding:15px 18px; margin:10px 0;'>
                    <b style='color:#5b21b6;'>Step 4 — Which models should we try?</b><br>
                    <span style='color:#555; font-size:0.92em;'>
                    Each model is a different "strategy" for learning patterns. We'll train all selected ones and compare their accuracy.
                    </span>
                </div>""", unsafe_allow_html=True)

                if problem_type == "Regression":
                    available_models = {
                        "Linear Regression":        "📏 Draws a straight line through the data — simple and fast. Best when the relationship is straightforward.",
                        "Random Forest":            "🌲 Builds many decision trees and combines them — handles complex patterns well.",
                        "Gradient Boosting":        "🚀 Learns from mistakes step by step — often the most accurate but slower.",
                        "Support Vector Regression":"🎯 Finds the best boundary in complex data — good for smaller datasets.",
                        "Ridge Regression":         "📏 Like Linear Regression but prevents overfitting — good when features are correlated.",
                        "Lasso Regression":         "✂️ Like Ridge but also removes unimportant features automatically."
                    }
                else:
                    available_models = {
                        "Logistic Regression":      "📏 Simple and fast — draws a line to separate categories. Easy to explain.",
                        "Random Forest":            "🌲 Many decision trees voting together — reliable and accurate.",
                        "Gradient Boosting":        "🚀 Learns from mistakes step by step — often highest accuracy.",
                        "Support Vector Machine":   "🎯 Finds the best boundary between categories — good for complex data.",
                        "Decision Tree":            "🌿 Makes decisions like a flowchart — very easy to understand and explain."
                    }

                selected_models = []
                for model_name, description in available_models.items():
                    if st.checkbox(f"**{model_name}** — {description}",
                                   value=model_name in ["Linear Regression", "Random Forest", "Logistic Regression"]):
                        selected_models.append(model_name)

                # ── Step 5 ──
                st.markdown("""
                <div style='background:#fff; border:1px solid #e0e7ff; border-radius:10px; padding:15px 18px; margin:10px 0;'>
                    <b style='color:#5b21b6;'>Step 5 — Training settings</b><br>
                    <span style='color:#555; font-size:0.92em;'>
                    • <b>Test Set Size:</b> How much data to hold back for testing (not used in training). 20% is a good default.<br>
                    • <b>Cross-Validation:</b> Tests the model multiple times on different splits — gives a more reliable accuracy score.
                    </span>
                </div>""", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5,
                                          help="20% means 80% of data trains the model, 20% tests it")
                    random_state = st.number_input("Random State", 0, 1000, 42,
                                                   help="Just a number to make results reproducible — keep it at 42")
                with col2:
                    cross_validation = st.checkbox("Use Cross-Validation ✅ (recommended)", value=True,
                                                   help="Tests the model multiple times for a more reliable score")
                    cv_folds = st.slider("CV Folds", 3, 10, 5) if cross_validation else 5
                
                # Train Models Button
                if st.button("🚀 Train Selected Models", disabled=not (feature_cols and selected_models)) and feature_cols and selected_models:
                    train_model_enhanced(target_col, feature_cols, selected_models, problem_type, test_size/100, random_state, cross_validation, cv_folds)