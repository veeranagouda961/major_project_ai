import os
import sys
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from flask_cors import CORS


# LangChain and AI Imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="langchain_google_genai")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")


from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from transformers import AutoformerForPrediction, AutoformerConfig
import torch

# Local chart utilities
try:
    from charts import generate_chart
except Exception:
    generate_chart = None

# Load environment variables from .env file
load_dotenv()

# Configure the API key globally


# --- Flask App Initialization ---
app = Flask(__name__)

# Load environment variables
cors_origin = os.getenv("CORS_ORIGIN", "http://localhost:8080")
flask_host = os.getenv("FLASK_HOST", "0.0.0.0")
flask_port = int(os.getenv("FLASK_PORT", "5000"))
flask_debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"

CORS(app, resources={r"/*": {"origins": cors_origin, "supports_credentials": True}})
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER", "uploads")
app.config['STATIC_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# --- Global Variables & Pre-loading ---
KNOWLEDGE_BASE_PATH = "knowledge_base"
FAISS_INDEX_PATH = "faiss_index"

llm = ChatGoogleGenerativeAI(
    model="gemini-pro-latest",
    temperature=0.1,
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    },
)

# Initialize embeddings with offline mode and error handling
try:
    # Try to load embeddings with offline mode first
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("SUCCESS: HuggingFace embeddings loaded successfully.")
except Exception as e:
    print(f"WARNING: Failed to load HuggingFace embeddings: {e}")
    print("INFO: Attempting to use offline mode...")
    try:
        # Try offline mode
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("SUCCESS: HuggingFace embeddings loaded in offline mode.")
    except Exception as e2:
        print(f"ERROR: Failed to load embeddings even in offline mode: {e2}")
        print("INFO: Application will run without knowledge base functionality.")
        embeddings = None

knowledge_chain = None

# --- Knowledge Base Pre-processing Function ---
def initialize_knowledge_base():
    """
    Loads documents from the knowledge_base, creates a vector store,
    and saves it to disk if it doesn't already exist.
    """
    global knowledge_chain
    
    # Check if embeddings are available
    if embeddings is None:
        print("WARNING: Embeddings not available. Knowledge base functionality disabled.")
        knowledge_chain = "Knowledge base unavailable - embeddings not loaded."
        return
    
    if os.path.exists(FAISS_INDEX_PATH):
        print("SUCCESS: Loading existing FAISS index for knowledge base.")
        try:
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"ERROR: Failed to load existing FAISS index: {e}")
            print("INFO: Will attempt to recreate index...")
            knowledge_chain = "Knowledge base unavailable - index loading failed."
            return
    else:
        print("INFO: Creating new FAISS index for knowledge base...")
        os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)
        pdf_loader = DirectoryLoader(KNOWLEDGE_BASE_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True)
        txt_loader = DirectoryLoader(KNOWLEDGE_BASE_PATH, glob="**/*.txt", loader_cls=TextLoader, recursive=True)
        documents = pdf_loader.load() + txt_loader.load()

        if not documents:
            print("WARNING: No documents found in the knowledge_base folder. Strategic advice will be limited.")
            knowledge_chain = "No knowledge base loaded."
            return

        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            texts = text_splitter.split_documents(documents)
            vector_store = FAISS.from_documents(texts, embeddings)
            vector_store.save_local(FAISS_INDEX_PATH)
            print(f"SUCCESS: Index created and saved to '{FAISS_INDEX_PATH}'.")
        except Exception as e:
            print(f"ERROR: Failed to create FAISS index: {e}")
            knowledge_chain = "Knowledge base unavailable - index creation failed."
            return

    try:
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        knowledge_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        print("SUCCESS: Virtual CFO Knowledge Base is ready.")
    except Exception as e:
        print(f"ERROR: Failed to create knowledge chain: {e}")
        knowledge_chain = "Knowledge base unavailable - chain creation failed."

# --- Helper Functions for AI Models ---
def find_csv_columns(csv_path):
    """Detect a likely date column and a primary numeric value column.

    Heuristics:
    - Prefer columns whose name contains 'date'/'time' and whose parse success rate > 80%.
    - Otherwise, evaluate all object-like columns and pick the one with the highest parse success.
    - Choose a numeric value column that is not an id/year-like field.
    """
    try:
        df_sample = pd.read_csv(csv_path, nrows=500)
        date_candidates = list(df_sample.columns)

        def parse_rate(series):
            parsed = pd.to_datetime(series, errors='coerce', utc=False, dayfirst=False, infer_datetime_format=True)
            return parsed.notna().mean()

        # Score columns by name hint and parse success
        best_col = None
        best_score = 0.0
        for col in date_candidates:
            # Only attempt on non-numeric columns to avoid mis-parsing numeric values as dates
            if pd.api.types.is_numeric_dtype(df_sample[col]):
                continue
            score = parse_rate(df_sample[col])
            name_bonus = 0.3 if any(k in col.lower() for k in ['date', 'time', 'day', 'month']) else 0.0
            total = score + name_bonus
            if total > best_score:
                best_score = total
                best_col = col

        date_col = best_col if best_col and best_score >= 0.6 else None

        numeric_cols = [c for c in df_sample.select_dtypes(include=[np.number]).columns
                        if 'id' not in c.lower() and 'year' not in c.lower() and 'month' not in c.lower()]
        value_col = numeric_cols[-1] if numeric_cols else None

        return date_col, value_col
    except Exception as e:
        print(f"Error analyzing CSV columns: {e}")
        return None, None

def detect_anomalies(csv_path, date_col, value_col):
    """Detect anomalies in a numeric column using the IQR method and plot them."""
    if not value_col:
        return "Could not identify a primary numeric column for anomaly detection.", None
    
    df = pd.read_csv(csv_path)
    if value_col not in df.columns:
        return f"Column '{value_col}' not found.", None

    Q1 = df[value_col].quantile(0.25)
    Q3 = df[value_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    anomalies = df[(df[value_col] < lower_bound) | (df[value_col] > upper_bound)]

    if anomalies.empty:
        return "No significant anomalies detected in the data.", None
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x=date_col if date_col and date_col in df.columns else df.index, y=value_col, label='Data')
    sns.scatterplot(data=anomalies, x=date_col if date_col and date_col in df.columns else anomalies.index, y=value_col, color='red', s=100, label='Anomalies')
    plt.title(f'Anomaly Detection for {value_col}')
    plt.xlabel(date_col if date_col else 'Index')
    plt.ylabel(value_col)
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(app.config['STATIC_FOLDER'], 'anomaly_plot.png')
    plt.savefig(plot_path)
    plt.close()

    summary = f"Detected {len(anomalies)} potential anomalies in '{value_col}'. These are values significantly lower than {lower_bound:.2f} or higher than {upper_bound:.2f}."
    return summary, f'/{plot_path}'

def format_response_with_bold_tags(text):
    """Format response text by converting markdown to HTML, removing ###, ***, and highlighting numbers."""
    import re
    
    # Remove ### headers and *** dividers
    text = re.sub(r'###\s*', '', text)
    text = re.sub(r'\*\*\*+', '', text)
    
    # Convert markdown bold to HTML bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # Convert markdown bullet points to hyphens
    text = re.sub(r'^\s*[*•]\s+', '- ', text, flags=re.MULTILINE)
    
    # Highlight all numbers (currency, percentages, regular numbers)
    text = re.sub(r'(₹[\d,]+\.?\d*)', r'<b>\1</b>', text)
    text = re.sub(r'(\$[\d,]+\.?\d*)', r'<b>\1</b>', text)
    text = re.sub(r'([\d,]+\.?\d*%)', r'<b>\1</b>', text)
    text = re.sub(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b', r'<b>\1</b>', text)
    
    return text

def predict_timeseries(csv_path, date_col, value_col, prediction_length=12):
    """Predict future values using a lightweight linear trend over recent data and generate a plot.

    Also returns a short natural-language explanation derived from the latest
    historical values and forecast trajectory so the frontend can describe the
    chart meaningfully.
    """
    if not date_col or not value_col:
        return "Could not identify suitable date and value columns for forecasting.", None
        
    # Read CSV and parse only the detected date column
    df = pd.read_csv(csv_path)
    use_index = False
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        if df.empty:
            use_index = True
        else:
            df = df.sort_values(by=date_col).reset_index(drop=True)
    else:
        use_index = True
    
    data = df[value_col].astype(float).values
    # Use a recent window to fit a simple linear trend
    recent_window = int(min(max(10, prediction_length * 2), len(data)))
    y_recent = data[-recent_window:]
    x_recent = np.arange(len(y_recent), dtype=float)
    try:
        slope, intercept = np.polyfit(x_recent, y_recent, 1)
    except Exception:
        slope, intercept = 0.0, float(y_recent[-1]) if len(y_recent) else 0.0

    x_future = np.arange(len(y_recent), len(y_recent) + prediction_length, dtype=float)
    mean_prediction = (slope * x_future + intercept).tolist()

    if not use_index:
        freq = pd.infer_freq(df[date_col]) if len(df[date_col]) > 2 else 'D'
        last_date = df[date_col].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=prediction_length + 1, freq=freq)[1:]
    else:
        # Fallback to simple index-based x-axis
        future_dates = np.arange(len(df), len(df) + prediction_length)

    plt.figure(figsize=(12, 6))
    if not use_index:
        plt.plot(df[date_col], df[value_col], label='Historical Data')
        plt.plot(future_dates, mean_prediction, label='Forecast', linestyle='--')
        plt.xlabel(date_col)
    else:
        plt.plot(np.arange(len(df)), df[value_col], label='Historical Data')
        plt.plot(future_dates, mean_prediction, label='Forecast', linestyle='--')
        plt.xlabel('Index')
    plt.title(f'Forecast for {value_col}')
    plt.ylabel(value_col)
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(app.config['STATIC_FOLDER'], 'forecast_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Build a concise, data-grounded explanation
    recent_window = min(len(df), max(6, prediction_length))
    recent_series = df[value_col].tail(recent_window)
    recent_change = (recent_series.iloc[-1] - recent_series.iloc[0]) if recent_window > 1 else 0
    recent_pct = (recent_change / recent_series.iloc[0] * 100.0) if recent_window > 1 and recent_series.iloc[0] != 0 else 0
    forecast_change = mean_prediction[-1] - (recent_series.iloc[-1] if len(recent_series) else 0)
    forecast_dir = "increase" if forecast_change > 0 else ("decrease" if forecast_change < 0 else "remain roughly flat")

    summary = (
        f"Forecast generated for the next {prediction_length} periods. "
        f"Recent trend: {recent_pct:.1f}% change over the last {recent_window} observations. "
        f"The projection suggests a {forecast_dir} toward the horizon. See the chart for details."
    )
    return summary, f'/{plot_path}'

def plot_rate_of_change(csv_path, date_col, value_col, two_month_window=True):
    """Compute daily percentage rate of change, plot it, optionally aggregate by 2-month windows,
    save to static and also export a copy to the project root as 'amazon_sales_roc.png'. Additionally,
    forecast the rate-of-change series and save a separate forecast plot.
    """
    if not value_col:
        return "Could not identify a numeric column for rate-of-change.", None, None

    df = pd.read_csv(csv_path)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(by=date_col)
        x = df[date_col]
    else:
        df = df.reset_index().rename(columns={'index': 'idx'})
        x = df['idx']

    y = pd.to_numeric(df[value_col], errors='coerce')
    s = pd.Series(y.values, index=x)

    # If index is datetime-like, ensure daily frequency for ROC
    if isinstance(s.index, pd.DatetimeIndex):
        s = s.asfreq('D')
        s = s.interpolate(limit_direction='both')
    roc = s.pct_change().mul(100.0)

    plt.figure(figsize=(12, 6))
    plt.plot(roc.index, roc.values, marker='o', linestyle='-', linewidth=1, markersize=2)
    plt.title(f'Daily Rate of Change in {value_col} (%)')
    plt.xlabel('Date' if isinstance(roc.index, pd.DatetimeIndex) else 'Index')
    plt.ylabel('Percentage Change (%)')
    plt.grid(True, alpha=0.3)

    # Optional two-month window smoothing/aggregation
    if two_month_window and isinstance(roc.index, pd.DatetimeIndex):
        two_m = roc.resample('2MS').mean()  # mean at each 2-month start
        plt.plot(two_m.index, two_m.values, color='orange', linewidth=2, label='2-month avg')
        plt.legend()

    roc_path_static = os.path.join(app.config['STATIC_FOLDER'], 'roc_plot.png')
    plt.savefig(roc_path_static)
    plt.close()

    # Save an additional export copy in project root as requested
    export_copy = os.path.join(os.getcwd(), 'amazon_sales_roc.png')
    try:
        shutil.copyfile(roc_path_static, export_copy)
    except Exception as _:
        pass

    # Forecast the ROC series using the same linear-trend method
    if isinstance(roc.index, pd.DatetimeIndex):
        clean = roc.dropna()
        if len(clean) >= 5:
            y_recent = clean.values[-min(60, len(clean)) :]
            x_recent = np.arange(len(y_recent), dtype=float)
            try:
                slope, intercept = np.polyfit(x_recent, y_recent, 1)
            except Exception:
                slope, intercept = 0.0, float(y_recent[-1])
            horizon = 14
            x_future = np.arange(len(y_recent), len(y_recent) + horizon, dtype=float)
            y_future = slope * x_future + intercept
            last_date = clean.index[-1]
            future_idx = pd.date_range(last_date, periods=horizon + 1, freq='D')[1:]

            plt.figure(figsize=(12, 6))
            plt.plot(clean.index, clean.values, label='ROC (historical)')
            plt.plot(future_idx, y_future, linestyle='--', label='ROC forecast')
            plt.title('Rate-of-Change Forecast (%)')
            plt.xlabel('Date')
            plt.ylabel('Percentage Change (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            roc_forecast_path = os.path.join(app.config['STATIC_FOLDER'], 'roc_forecast_plot.png')
            plt.savefig(roc_forecast_path)
            plt.close()
        else:
            roc_forecast_path = None
    else:
        roc_forecast_path = None

    summary = (
        "Computed daily percentage rate of change and generated the plot. "
        "A 2-month average line is included for smoother trends. An export copy was saved as 'amazon_sales_roc.png'."
    )
    return summary, f'/{roc_path_static}', (f'/{roc_forecast_path}' if roc_forecast_path else None)

def plot_linear_relationships(csv_path, date_col):
    """Find strong linear relations between numeric columns and plot them in subplots.
    We compute Pearson correlations and plot the top correlated pairs side-by-side.
    """
    df = pd.read_csv(csv_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return "Not enough numeric columns to assess linear relations.", None

    corr = df[numeric_cols].corr(method='pearson').abs()
    pairs = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i+1:]:
            pairs.append(((a, b), corr.loc[a, b]))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = [p for p in pairs if np.isfinite(p[1])][:6]
    if not top_pairs:
        return "No clear linear relations found between numeric columns.", None

    n = len(top_pairs)
    rows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4*rows))
    axes = np.array(axes).reshape(-1)
    for idx, ((a, b), r) in enumerate(top_pairs):
        ax = axes[idx]
        ax.plot(df[a], label=a)
        ax.plot(df[b], label=b)
        ax.set_title(f"{a} vs {b} (|r|={r:.2f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    out_path = os.path.join(app.config['STATIC_FOLDER'], 'linear_relations.png')
    plt.savefig(out_path)
    plt.close()
    return "Plotted top linear relations across numeric columns.", f'/{out_path}'

def plot_top_sales_channels(csv_path, date_col, value_col):
    """Detect a categorical 'channel' column and plot top 5 by total value_col."""
    df = pd.read_csv(csv_path)
    if not value_col or value_col not in df.columns:
        return "Could not identify a numeric value column for sales.", None
    # Find a categorical column with reasonable cardinality
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    best_cat = None
    best_card = None
    for c in categorical_cols:
        unique = df[c].nunique(dropna=True)
        if 2 <= unique <= 20 and (best_card is None or unique < best_card):
            best_cat, best_card = c, unique
        if 'channel' in c.lower():
            best_cat = c
            break
    if not best_cat:
        return "No suitable categorical column found for channels.", None
    grouped = df.groupby(best_cat)[value_col].sum().sort_values(ascending=False).head(5)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=grouped.values, y=grouped.index, orient='h')
    plt.title('Top 5 Sales Channels')
    plt.xlabel(value_col)
    plt.ylabel(best_cat)
    plt.grid(True, axis='x', alpha=0.2)
    out_path = os.path.join(app.config['STATIC_FOLDER'], 'top_channels.png')
    plt.savefig(out_path)
    plt.close()
    return f"Top 5 '{best_cat}' by total {value_col}.", f'/{out_path}'

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files: return "No file part", 400
        file = request.files['file']
        if file.filename == '': return "No selected file", 400
        if file and file.filename.lower().endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            session['csv_path'] = filepath
            return render_template('index.html', file_uploaded=True, filename=file.filename)
    return render_template('index.html', file_uploaded=False)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

@app.route('/chat', methods=['POST'])
def chat():
    user_prompt = request.json.get("prompt", "").lower()
    csv_path = session.get('csv_path')
    response_data = {}

    if not user_prompt: return jsonify({"error": "No prompt provided."}), 400
    if not csv_path or not os.path.exists(csv_path): return jsonify({"error": "CSV file not found. Please upload a file first."}), 400

    try:
        date_col, value_col = find_csv_columns(csv_path)
        final_response_text = ""
        
        # --- Keyword-based Task Router ---
        
        # 1. Handle special tasks first
        # Rate-of-change / growth requests
        if any(k in user_prompt for k in ["rate of change", "roc", "growth rate", "percentage change"]):
            summary, roc_url, roc_forecast_url = plot_rate_of_change(csv_path, date_col, value_col, two_month_window=True)
            if roc_url: response_data['image_url'] = roc_url
            if roc_forecast_url: response_data['secondary_image_url'] = roc_forecast_url
            final_response_text = summary

        # Linear relationships across numeric columns
        elif any(k in user_prompt for k in ["linear relation", "linear relationship", "correlation", "sub plots", "subplots"]):
            summary, img_url = plot_linear_relationships(csv_path, date_col)
            if img_url: response_data['image_url'] = img_url
            final_response_text = summary

        # Top sales channels
        elif any(k in user_prompt for k in ["top 5", "top five", "best sales channel", "top sales channel", "top channels"]):
            summary, img_url = plot_top_sales_channels(csv_path, date_col, value_col)
            if img_url: response_data['image_url'] = img_url
            final_response_text = summary

        # Chart requests - check if user wants explanation with chart
        elif "pie" in user_prompt and ("chart" in user_prompt or "graph" in user_prompt or "plot" in user_prompt):
            if generate_chart is None:
                final_response_text = "Chart generator unavailable."
            else:
                msg, img_url = generate_chart('pie', csv_path, date_col, value_col)
                if img_url: response_data['image_url'] = img_url
                
                # Check if user wants explanation
                if any(word in user_prompt for word in ['explain', 'tell me', 'what', 'why', 'how', 'analyze', 'insight']):
                    explanation = ""
                    if knowledge_chain and not isinstance(knowledge_chain, str):
                        try:
                            kb_query = f"Provide concise insights about pie chart analysis for {value_col or 'financial metrics'}. Focus on practical CFO-level interpretation. Keep under 100 words."
                            kb_res = knowledge_chain.invoke({"query": kb_query})
                            explanation = kb_res.get('result', '')
                        except Exception:
                            pass
                    final_response_text = msg + ("\n\n" + explanation if explanation else "")
                else:
                    final_response_text = msg
        
        elif "bar" in user_prompt and ("chart" in user_prompt or "graph" in user_prompt or "plot" in user_prompt) and "stacked" not in user_prompt:
            if generate_chart is None:
                final_response_text = "Chart generator unavailable."
            else:
                msg, img_url = generate_chart('bar', csv_path, date_col, value_col)
                if img_url: response_data['image_url'] = img_url
                
                # Check if user wants explanation
                if any(word in user_prompt for word in ['explain', 'tell me', 'what', 'why', 'how', 'analyze', 'insight']):
                    explanation = ""
                    if knowledge_chain and not isinstance(knowledge_chain, str):
                        try:
                            kb_query = f"Provide concise insights about bar chart analysis for {value_col or 'financial metrics'}. Focus on practical CFO-level interpretation. Keep under 100 words."
                            kb_res = knowledge_chain.invoke({"query": kb_query})
                            explanation = kb_res.get('result', '')
                        except Exception:
                            pass
                    final_response_text = msg + ("\n\n" + explanation if explanation else "")
                else:
                    final_response_text = msg
        
        elif "line" in user_prompt and ("chart" in user_prompt or "graph" in user_prompt or "plot" in user_prompt):
            if generate_chart is None:
                final_response_text = "Chart generator unavailable."
            else:
                msg, img_url = generate_chart('line', csv_path, date_col, value_col)
                if img_url: response_data['image_url'] = img_url
                
                # Check if user wants explanation
                if any(word in user_prompt for word in ['explain', 'tell me', 'what', 'why', 'how', 'analyze', 'insight']):
                    explanation = ""
                    if knowledge_chain and not isinstance(knowledge_chain, str):
                        try:
                            kb_query = f"Provide concise insights about line chart trend analysis for {value_col or 'financial metrics'}. Focus on practical CFO-level interpretation. Keep under 100 words."
                            kb_res = knowledge_chain.invoke({"query": kb_query})
                            explanation = kb_res.get('result', '')
                        except Exception:
                            pass
                    final_response_text = msg + ("\n\n" + explanation if explanation else "")
                else:
                    final_response_text = msg
        
        elif "area" in user_prompt and ("chart" in user_prompt or "graph" in user_prompt or "plot" in user_prompt):
            if generate_chart is None:
                final_response_text = "Chart generator unavailable."
            else:
                msg, img_url = generate_chart('area', csv_path, date_col, value_col)
                if img_url: response_data['image_url'] = img_url
                
                # Check if user wants explanation
                if any(word in user_prompt for word in ['explain', 'tell me', 'what', 'why', 'how', 'analyze', 'insight']):
                    explanation = ""
                    if knowledge_chain and not isinstance(knowledge_chain, str):
                        try:
                            kb_query = f"Provide concise insights about area chart analysis for {value_col or 'financial metrics'}. Focus on practical CFO-level interpretation. Keep under 100 words."
                            kb_res = knowledge_chain.invoke({"query": kb_query})
                            explanation = kb_res.get('result', '')
                        except Exception:
                            pass
                    final_response_text = msg + ("\n\n" + explanation if explanation else "")
                else:
                    final_response_text = msg
        
        elif "scatter" in user_prompt and ("chart" in user_prompt or "graph" in user_prompt or "plot" in user_prompt):
            if generate_chart is None:
                final_response_text = "Chart generator unavailable."
            else:
                msg, img_url = generate_chart('scatter', csv_path, date_col, value_col)
                if img_url: response_data['image_url'] = img_url
                
                # Check if user wants explanation
                if any(word in user_prompt for word in ['explain', 'tell me', 'what', 'why', 'how', 'analyze', 'insight']):
                    explanation = ""
                    if knowledge_chain and not isinstance(knowledge_chain, str):
                        try:
                            kb_query = f"Provide concise insights about scatter plot correlation analysis. Focus on practical CFO-level interpretation. Keep under 100 words."
                            kb_res = knowledge_chain.invoke({"query": kb_query})
                            explanation = kb_res.get('result', '')
                        except Exception:
                            pass
                    final_response_text = msg + ("\n\n" + explanation if explanation else "")
                else:
                    final_response_text = msg
        
        elif "box" in user_prompt and ("chart" in user_prompt or "graph" in user_prompt or "plot" in user_prompt):
            if generate_chart is None:
                final_response_text = "Chart generator unavailable."
            else:
                msg, img_url = generate_chart('box', csv_path, date_col, value_col)
                if img_url: response_data['image_url'] = img_url
                
                # Check if user wants explanation
                if any(word in user_prompt for word in ['explain', 'tell me', 'what', 'why', 'how', 'analyze', 'insight']):
                    explanation = ""
                    if knowledge_chain and not isinstance(knowledge_chain, str):
                        try:
                            kb_query = f"Provide concise insights about box plot distribution analysis. Focus on practical CFO-level interpretation. Keep under 100 words."
                            kb_res = knowledge_chain.invoke({"query": kb_query})
                            explanation = kb_res.get('result', '')
                        except Exception:
                            pass
                    final_response_text = msg + ("\n\n" + explanation if explanation else "")
                else:
                    final_response_text = msg
        
        elif ("heatmap" in user_prompt or "heat map" in user_prompt) and ("chart" in user_prompt or "graph" in user_prompt or "plot" in user_prompt or "correlation" in user_prompt):
            if generate_chart is None:
                final_response_text = "Chart generator unavailable."
            else:
                msg, img_url = generate_chart('heatmap', csv_path, date_col, value_col)
                if img_url: response_data['image_url'] = img_url
                
                # Check if user wants explanation
                if any(word in user_prompt for word in ['explain', 'tell me', 'what', 'why', 'how', 'analyze', 'insight']):
                    explanation = ""
                    if knowledge_chain and not isinstance(knowledge_chain, str):
                        try:
                            kb_query = f"Provide concise insights about correlation heatmap analysis. Focus on practical CFO-level interpretation. Keep under 100 words."
                            kb_res = knowledge_chain.invoke({"query": kb_query})
                            explanation = kb_res.get('result', '')
                        except Exception:
                            pass
                    final_response_text = msg + ("\n\n" + explanation if explanation else "")
                else:
                    final_response_text = msg
        
        elif "waterfall" in user_prompt and ("chart" in user_prompt or "graph" in user_prompt or "plot" in user_prompt):
            if generate_chart is None:
                final_response_text = "Chart generator unavailable."
            else:
                msg, img_url = generate_chart('waterfall', csv_path, date_col, value_col)
                if img_url: response_data['image_url'] = img_url
                
                # Check if user wants explanation
                if any(word in user_prompt for word in ['explain', 'tell me', 'what', 'why', 'how', 'analyze', 'insight']):
                    explanation = ""
                    if knowledge_chain and not isinstance(knowledge_chain, str):
                        try:
                            kb_query = f"Provide concise insights about waterfall chart financial breakdown analysis. Focus on practical CFO-level interpretation. Keep under 100 words."
                            kb_res = knowledge_chain.invoke({"query": kb_query})
                            explanation = kb_res.get('result', '')
                        except Exception:
                            pass
                    final_response_text = msg + ("\n\n" + explanation if explanation else "")
                else:
                    final_response_text = msg

        # Forecast/predict requests
        elif "forecast" in user_prompt or "predict" in user_prompt:
            summary, plot_url = predict_timeseries(csv_path, date_col, value_col)
            if plot_url: response_data['image_url'] = plot_url
            final_response_text = summary
        
        # Generic chart/graph/plot requests - default to asking user to be specific
        elif "chart" in user_prompt or "graph" in user_prompt or "plot" in user_prompt or "compare" in user_prompt:
            final_response_text = "I can generate various types of charts for you. Please specify which type you'd like:\n\n" + \
                                  "- **Pie chart** - for showing proportions and percentages\n" + \
                                  "- **Bar chart** - for comparing categories\n" + \
                                  "- **Line chart** - for showing trends over time\n" + \
                                  "- **Area chart** - for cumulative trends\n" + \
                                  "- **Scatter plot** - for showing relationships\n" + \
                                  "- **Box plot** - for distribution analysis\n" + \
                                  "- **Heatmap** - for correlation analysis\n" + \
                                  "- **Waterfall chart** - for financial breakdown\n\n" + \
                                  "Or you can ask for a **forecast** to predict future trends."
            
        elif "anomaly" in user_prompt or "outlier" in user_prompt:
            summary, plot_url = detect_anomalies(csv_path, date_col, value_col)
            if plot_url: response_data['image_url'] = plot_url
            final_response_text = summary
        
        else:
            # ALWAYS ANALYZE DATASET FIRST, THEN ADD KNOWLEDGE BASE INSIGHTS
            print("➡️ Analyzing dataset first, then adding knowledge base insights...")
            
            # Get data insights first using CSV agent
            print("➡️ Analyzing dataset...")
            data_agent_prompt = f"""
            Analyze the financial dataset to answer: '{user_prompt}'
            
            INSTRUCTIONS:
            1. Extract relevant data points related to the user's question
            2. Calculate key metrics (totals, averages, trends, etc.)
            3. Provide specific numbers and insights from the dataset
            4. Be concise and to-the-point - avoid long explanations
            5. Use bullet points with hyphens (-) for listing items
            6. Your response MUST start with "Final Answer:"
            7. Be specific with numbers, dates, and amounts
            8. Always base your answer on the actual data in the CSV file
            9. Keep response under 200 words
            
            User's question: {user_prompt}
            """

            csv_agent = create_csv_agent(
                llm,
                csv_path,
                verbose=False,
                allow_dangerous_code=True,
                agent_executor_kwargs={
                    "handle_parsing_errors": True
                },
                max_iterations=30,
                max_execution_time=120
            )

            data_insights = ""
            try:
                data_result = csv_agent.invoke({"input": data_agent_prompt})
                data_insights = data_result.get('output', "")
                
                if "Final Answer:" in data_insights:
                    data_insights = data_insights.split("Final Answer:")[-1].strip()
                    
            except Exception as agent_error:
                print(f"Data analysis failed: {agent_error}")
                data_insights = f"Unable to analyze dataset: {str(agent_error)}"
            
            # Get strategic advice from knowledge base ONLY if relevant
            strategic_advice = ""
            if knowledge_chain and not isinstance(knowledge_chain, str):
                # Only query knowledge base for strategic/improvement questions
                strategic_keywords = ['improve', 'strategy', 'recommendation', 'advice', 'how to', 'what should', 'best practice', 'optimize', 'increase', 'decrease', 'reduce', 'grow', 'turnaround']
                if any(keyword in user_prompt for keyword in strategic_keywords):
                    print("➡️ Getting strategic advice from knowledge base...")
                    try:
                        kb_prompt = f"""
                        Based on the user's question: '{user_prompt}'
                        And the data context: {data_insights[:300] if data_insights else 'N/A'}
                        
                        Provide concise, actionable CFO-level recommendations.
                        - Keep response under 150 words
                        - Use bullet points with hyphens (-)
                        - Focus on practical actions
                        - Avoid generic advice
                        """
                        strategy_result = knowledge_chain.invoke({"query": kb_prompt})
                        strategic_advice = strategy_result['result']
                    except Exception as e:
                        print(f"Knowledge base query failed: {e}")
                        strategic_advice = ""
            
            # Combine insights concisely
            if data_insights and strategic_advice:
                final_response_text = f"{data_insights}\n\n<b>Recommendations:</b>\n{strategic_advice}"
            elif data_insights:
                final_response_text = data_insights
            elif strategic_advice:
                final_response_text = strategic_advice
            else:
                final_response_text = "I need more context to provide a helpful analysis. Could you please be more specific about what you'd like to know?"

        # Format response with bold tags for better presentation
        final_response_text = format_response_with_bold_tags(final_response_text)
        
        response_data['response'] = final_response_text
        return jsonify(response_data)

    except Exception as e:
        print(f"Error during chat processing: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- Main Application Execution ---
if __name__ == '__main__':
    if '--rebuild' in sys.argv:
        if os.path.exists(FAISS_INDEX_PATH):
            print(f"INFO: '--rebuild' flag detected. Deleting existing index '{FAISS_INDEX_PATH}'...")
            shutil.rmtree(FAISS_INDEX_PATH)
            print("SUCCESS: Index deleted.")
    
    initialize_knowledge_base()
    
    # Use environment variables for Flask configuration
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    print(f"Starting Flask server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)