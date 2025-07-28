import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="BERT Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .token-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .prediction-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        # Check if model files exist
        model_path = './saved_model'
        tokenizer_path = './tokenizer'
        
        if not os.path.exists(model_path):
            st.error(f"Model not found at {model_path}. Please run the training notebook first.")
            return None, None
            
        if not os.path.exists(tokenizer_path):
            st.error(f"Tokenizer not found at {tokenizer_path}. Please run the training notebook first.")
            return None, None
        
        # Load model and tokenizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        model.to(device)
        model.eval()
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def tokenize_and_visualize(text, tokenizer, max_length=128):
    """Tokenize text and return visualization data"""
    # Basic tokenization
    tokens = tokenizer.tokenize(text)
    
    # Full encoding
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Get token IDs and attention mask
    input_ids = encoding['input_ids'][0].numpy()
    attention_mask = encoding['attention_mask'][0].numpy()
    
    # Convert IDs back to tokens for visualization
    tokens_with_special = tokenizer.convert_ids_to_tokens(input_ids)
    
    return {
        'original_tokens': tokens,
        'tokens_with_special': tokens_with_special,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'encoding': encoding
    }

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for given text"""
    # Encode the text
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1)
    
    return predicted_class.item(), predictions.cpu().numpy()[0]

def create_token_visualization(tokenization_data):
    """Create visualization for tokens"""
    tokens = tokenization_data['tokens_with_special']
    input_ids = tokenization_data['input_ids']
    attention_mask = tokenization_data['attention_mask']
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        'Position': range(len(tokens)),
        'Token': tokens,
        'Input_ID': input_ids,
        'Attention_Mask': attention_mask
    })
    
    # Filter only attended tokens for cleaner visualization
    attended_df = df[df['Attention_Mask'] == 1]
    
    return df, attended_df

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 BERT Text Classification System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Tokenization, Encoding, and Sentiment Analysis with BERT</p>', unsafe_allow_html=True)
    
    # Load model and tokenizer
    model_data = load_model_and_tokenizer()
    if model_data[0] is None:
        st.stop()
    
    model, tokenizer, device = model_data
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Configuration")
    max_length = st.sidebar.slider("Max Sequence Length", 32, 512, 128)
    show_special_tokens = st.sidebar.checkbox("Show Special Tokens ([CLS], [SEP], [PAD])", True)
    show_token_ids = st.sidebar.checkbox("Show Token IDs", False)
    
    st.sidebar.markdown("## 📊 Model Info")
    st.sidebar.info(f"Device: {device}")
    st.sidebar.info("Model: BERT-base-uncased")
    
    # Load training history if available
    if os.path.exists('training_history.pkl'):
        with open('training_history.pkl', 'rb') as f:
            history = pickle.load(f)
        st.sidebar.success(f"Best Accuracy: {history['best_accuracy']:.3f}")
        st.sidebar.success(f"Test Accuracy: {history['test_accuracy']:.3f}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📝 Text Input</h2>', unsafe_allow_html=True)
        
        # Text input options
        input_option = st.radio("Choose input method:", ["Type text", "Select example"])
        
        if input_option == "Type text":
            user_text = st.text_area(
                "Enter your text for analysis:",
                value="This movie is absolutely fantastic! Great acting and storyline.",
                height=100,
                help="Enter any text to see how BERT tokenizes and classifies it"
            )
        else:
            example_texts = [
                "This movie is absolutely fantastic! Great acting and storyline.",
                "I loved every minute of this film. Highly recommended!",
                "This movie was terrible. Waste of time and money.",
                "Boring storyline with poor acting throughout.",
                "The acting was decent but the plot was confusing.",
                "An amazing masterpiece with incredible performances!"
            ]
            user_text = st.selectbox("Select an example:", example_texts)
    
    with col2:
        st.markdown('<h2 class="sub-header">🎯 Quick Stats</h2>', unsafe_allow_html=True)
        if user_text:
            word_count = len(user_text.split())
            char_count = len(user_text)
            st.metric("Words", word_count)
            st.metric("Characters", char_count)
    
    if user_text and st.button("🔍 Analyze Text", type="primary"):
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🔤 Tokenization", "🔢 Encoding", "🎯 Prediction", "📊 Visualization"])
        
        # Tokenize the text
        tokenization_data = tokenize_and_visualize(user_text, tokenizer, max_length)
        
        with tab1:
            st.markdown('<h3 class="sub-header">Tokenization Process</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Text:**")
                st.markdown(f'<div class="token-box">{user_text}</div>', unsafe_allow_html=True)
                
                st.markdown("**Basic Tokens:**")
                tokens_display = " | ".join([f"`{token}`" for token in tokenization_data['original_tokens']])
                st.markdown(tokens_display)
            
            with col2:
                if show_special_tokens:
                    st.markdown("**Tokens with Special Tokens:**")
                    special_tokens_display = " | ".join([f"`{token}`" for token in tokenization_data['tokens_with_special'] if token != '[PAD]'])
                    st.markdown(special_tokens_display)
                
                st.markdown("**Token Count:**")
                st.metric("Number of Tokens", len(tokenization_data['original_tokens']))
                st.metric("Total Sequence Length", len(tokenization_data['tokens_with_special']))
        
        with tab2:
            st.markdown('<h3 class="sub-header">Encoding Process</h3>', unsafe_allow_html=True)
            
            df, attended_df = create_token_visualization(tokenization_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Input IDs (Attended Tokens Only):**")
                if show_token_ids:
                    display_df = attended_df[['Token', 'Input_ID']].copy()
                else:
                    display_df = attended_df[['Token']].copy()
                st.dataframe(display_df, use_container_width=True)
            
            with col2:
                st.markdown("**Encoding Statistics:**")
                st.metric("Attended Tokens", int(attended_df.shape[0]))
                st.metric("Padded Tokens", int((df['Attention_Mask'] == 0).sum()))
                st.metric("Max Token ID", int(tokenization_data['input_ids'].max()))
                
                # Attention mask visualization
                attention_viz = go.Figure(data=go.Bar(
                    x=list(range(len(tokenization_data['attention_mask']))), 
                    y=tokenization_data['attention_mask'],
                    name='Attention Mask'
                ))
                attention_viz.update_layout(
                    title="Attention Mask",
                    xaxis_title="Position",
                    yaxis_title="Attention",
                    height=300
                )
                st.plotly_chart(attention_viz, use_container_width=True)
        
        with tab3:
            st.markdown('<h3 class="sub-header">Sentiment Prediction</h3>', unsafe_allow_html=True)
            
            # Make prediction
            predicted_class, probabilities = predict_sentiment(user_text, model, tokenizer, device)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Display prediction
                if predicted_class == 1:
                    st.markdown(f'''
                    <div class="prediction-positive">
                        <h4>🎉 Positive Sentiment</h4>
                        <p>Confidence: {probabilities[1]:.3f}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="prediction-negative">
                        <h4>😞 Negative Sentiment</h4>
                        <p>Confidence: {probabilities[0]:.3f}</p>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with col2:
                # Probability visualization
                prob_fig = go.Figure(data=[
                    go.Bar(x=['Negative', 'Positive'], y=probabilities, 
                           marker_color=['#ff4444', '#44ff44'])
                ])
                prob_fig.update_layout(
                    title="Prediction Probabilities",
                    yaxis_title="Probability",
                    height=300
                )
                st.plotly_chart(prob_fig, use_container_width=True)
        
        with tab4:
            st.markdown('<h3 class="sub-header">Token Analysis Visualization</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Token length distribution
                token_lengths = [len(token) for token in tokenization_data['original_tokens']]
                length_fig = px.histogram(
                    x=token_lengths, 
                    title="Token Length Distribution",
                    labels={'x': 'Token Length', 'y': 'Count'}
                 )
                st.plotly_chart(length_fig , use_container_width=True)
            
            with col2:
                # Token ID distribution
                attended_ids = tokenization_data['input_ids'][tokenization_data['attention_mask'] == 1]
                id_fig = px.scatter(
                    x=range(len(attended_ids)), 
                    y=attended_ids,
                    title="Token IDs by Position",
                    labels={'x': 'Position', 'y': 'Token ID'}
                )
                st.plotly_chart(id_fig, use_container_width=True)
            
            # Full token table
            st.markdown("**Complete Token Analysis:**")
            full_df = df.copy()
            if not show_special_tokens:
                full_df = full_df[full_df['Token'] != '[PAD]']
            
            st.dataframe(full_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🔍 How it works:")
    
    with st.expander("Understanding BERT Tokenization and Encoding"):
        st.markdown("""
        **1. Tokenization:**
        - Breaks text into subword tokens using WordPiece algorithm
        - Handles out-of-vocabulary words by splitting into smaller pieces
        - Adds special tokens: [CLS] at start, [SEP] at end
        
        **2. Encoding:**
        - Converts tokens to numerical IDs from BERT's vocabulary
        - Creates attention masks (1 for real tokens, 0 for padding)
        - Pads sequences to consistent length for batch processing
        
        **3. Model Processing:**
        - BERT processes the encoded sequences
        - Outputs contextualized representations for each token
        - Classification head makes final sentiment prediction
        
        **4. Key Features:**
        - Pre-trained on large text corpus
        - Bidirectional context understanding
        - Transfer learning for downstream tasks
        """)
    
    with st.expander("Model Architecture Details"):
        st.markdown("""
        **BERT Model Configuration:**
        - **Model:** bert-base-uncased
        - **Layers:** 12 transformer layers
        - **Hidden Size:** 768
        - **Attention Heads:** 12
        - **Vocabulary Size:** 30,522 tokens
        - **Max Sequence Length:** 512 tokens
        
        **Fine-tuning Details:**
        - **Task:** Binary sentiment classification
        - **Learning Rate:** 2e-5
        - **Batch Size:** 16
        - **Epochs:** 3
        - **Optimizer:** AdamW with linear warmup
        """)

if __name__ == "__main__":
    main()