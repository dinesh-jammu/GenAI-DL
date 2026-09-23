import json
import os
import pickle
import re
import string
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

MODEL_PATH_KERAS = "sentiment_lstm.keras"
MODEL_PATH_H5 = "sentiment_lstm.h5"
TOKENIZER_PATH = "tokenizer.pkl"
CONFIG_PATH = "config.json"

st.set_page_config(
    page_title="Sentiment AI | Movie Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high quality design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-weight: 700;
        font-size: 2.4rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        font-size: 1.1rem;
        color: #e0e7ff;
        margin: 0;
    }

    .result-card {
        padding: 1.8rem;
        border-radius: 16px;
        margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    
    .positive-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid #10b981;
    }
    
    .negative-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid #ef4444;
    }
    
    .sentiment-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.3rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .badge-pos {
        background-color: #10b981;
        color: white;
    }
    
    .badge-neg {
        background-color: #ef4444;
        color: white;
    }

    .audit-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }
    
    .audit-table th, .audit-table td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading deep learning model & tokenizer...")
def load_artifacts():
    model_path = MODEL_PATH_KERAS if os.path.exists(MODEL_PATH_KERAS) else MODEL_PATH_H5
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH_KERAS} or {MODEL_PATH_H5}")
    
    model = load_model(model_path)
    
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer file not found at {TOKENIZER_PATH}")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
        
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        
    return model, tokenizer, config


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(text: str, model, tokenizer, max_len: int):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    # PRE-padding avoids state corruption in short sequences
    padded = pad_sequences(seq, maxlen=max_len, padding="pre", truncating="pre")
    
    prob = float(model.predict(padded, verbose=0)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1.0 - prob
    return label, confidence, prob, cleaned, seq[0]


def main():
    # Top Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 IMDb Movie Review Sentiment AI</h1>
        <p>Bidirectional LSTM Neural Network for Real-time Sentiment Analysis & Realtime Audit</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load artifacts
    try:
        model, tokenizer, config = load_artifacts()
    except Exception as e:
        st.warning("⚠️ Model artifacts are not found in the workspace directory.")
        st.info("Run `python train.py` in your terminal to generate model artifacts.")
        st.error(f"Error details: {str(e)}")
        return

    # Sidebar Information
    with st.sidebar:
        st.title("⚙️ Model Details")
        st.markdown("---")
        
        st.subheader("📊 Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Accuracy", f"{config.get('test_accuracy', 0.0)*100:.1f}%")
        with col2:
            st.metric("ROC-AUC", f"{config.get('test_auc', 0.0):.4f}")
            
        st.markdown("---")
        st.subheader("🧠 Architecture Info")
        st.write(f"**Model:** Stacked Bidirectional LSTM")
        st.write(f"**Vocabulary Size:** {config.get('vocab_size', 15000):,} words")
        st.write(f"**Max Sequence Length:** {config.get('max_len', 200)} tokens")
        st.write(f"**Sequence Padding:** `Pre-padding` (Optimized for short reviews)")
        st.write(f"**Embedding Dim:** {config.get('embedding_dim', 128)}")
        st.write(f"**LSTM Units:** {config.get('lstm_units', 64)}")
        
        st.markdown("---")
        st.caption("🚀 Powered by TensorFlow & Streamlit")

    # App Tabs
    tab1, tab2 = st.tabs(["💬 Realtime Review Predictor", "🔬 Model Realtime Performance Audit"])
    
    with tab1:
        col_left, col_right = st.columns([7, 5])
        
        with col_left:
            st.subheader("📝 Enter Movie Review")
            
            presets = {
                "Select an example...": "",
                "⭐ Highly Positive Review": "This movie was an absolute masterpiece! The acting was top-notch, the score was thrilling, and the cinematography left me breathless.",
                "🚫 Highly Negative Review": "What a complete waste of time. The plot was full of plot holes, the dialogue was cringe-worthy, and the actors looked bored.",
                "🤔 Short Contrastive Review": "I expected it to be bad, but it turned out to be surprisingly amazing!",
                "⚡ Short Phrase": "This movie was great and I loved it!",
            }
            
            selected_preset = st.selectbox("Quick Presets:", list(presets.keys()))
            default_val = presets[selected_preset] if selected_preset != "Select an example..." else ""
            
            review_input = st.text_area(
                "Type or paste your review below:",
                value=default_val,
                height=180,
                placeholder="Write a movie review here (e.g., 'The plot was engaging and the acting was stellar...')"
            )
            
            col_btn, col_stats = st.columns([1, 2])
            with col_btn:
                analyze_clicked = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
                
            with col_stats:
                word_count = len(review_input.split()) if review_input.strip() else 0
                char_count = len(review_input)
                st.caption(f"**Text statistics:** {word_count} words | {char_count} characters")

        with col_right:
            st.subheader("📈 Prediction & Analytics")
            
            if analyze_clicked or (review_input.strip() and selected_preset != "Select an example..."):
                if not review_input.strip():
                    st.warning("Please enter a review first.")
                else:
                    with st.spinner("Analyzing text with LSTM..."):
                        label, confidence, prob, cleaned_text, tokens = predict_sentiment(
                            review_input, model, tokenizer, config.get("max_len", 200)
                        )
                        
                    is_pos = label == "Positive"
                    card_class = "positive-card" if is_pos else "negative-card"
                    badge_class = "badge-pos" if is_pos else "badge-neg"
                    emoji = "🎉" if is_pos else "👎"
                    
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <span class="sentiment-badge {badge_class}">{emoji} {label}</span>
                        <h2 style="margin-top: 1rem; font-size: 2.2rem; font-weight: 700; color: {'#10b981' if is_pos else '#ef4444'};">
                            {confidence * 100:.1f}% <span style="font-size: 1.1rem; color: #9ca3af;">Confidence</span>
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.write("**Probability breakdown:**")
                    st.caption(f"Positive Class Probability: `{prob:.4f}` | Negative Class Probability: `{1.0 - prob:.4f}`")
                    
                    st.progress(prob)
                    
                    with st.expander("🔍 Text Processing & Tokenization Details"):
                        st.write("**Cleaned input text:**")
                        st.code(cleaned_text if cleaned_text else "[Empty after cleaning]", language="text")
                        st.write(f"**First 20 Token IDs:**")
                        st.code(str(tokens[:20]), language="text")

            else:
                st.info("👈 Enter a review on the left and click **Analyze Sentiment** to see the prediction results.")

    with tab2:
        st.subheader("🧪 Real-world Benchmark & Diagnostic Audit")
        st.write("""
        Evaluating real-world reviews tests how well the LSTM model generalizes beyond full-length training reviews to short real-time user inputs.
        Click **Run Diagnostic Audit** to evaluate 10 benchmark real-time reviews.
        """)
        
        if st.button("▶️ Run Realtime Diagnostic Audit"):
            benchmark_suite = [
                ("This movie was great and I loved it!", "Positive"),
                ("This movie was terrible, awful and I hated it.", "Negative"),
                ("The movie was not good at all.", "Negative"),
                ("Not bad, actually quite enjoyable.", "Positive"),
                ("I expected it to be bad but it was amazing!", "Positive"),
                ("It was bad.", "Negative"),
                ("It was good.", "Positive"),
                ("Masterpiece of modern cinema.", "Positive"),
                ("Boring and slow, fell asleep after 10 minutes.", "Negative"),
                ("The acting was terrible but the visuals were great.", "Mixed/Positive"),
            ]
            
            audit_results = []
            for text, expected in benchmark_suite:
                label, conf, prob, clean, tokens = predict_sentiment(text, model, tokenizer, config.get("max_len", 200))
                audit_results.append({
                    "Review Input": text,
                    "Expected": expected,
                    "Model Output": label,
                    "Probability (Positive)": f"{prob:.4f}",
                    "Status": "✅ Pass" if (expected in label or label in expected) else "⚠️ Low Confidence/Mismatch"
                })
                
            st.dataframe(audit_results, use_container_width=True)
            st.success("Diagnostic Audit Completed!")

if __name__ == "__main__":
    main()
