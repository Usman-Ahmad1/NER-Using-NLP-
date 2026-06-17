import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
from pathlib import Path

# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="NER Entity Extractor",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for better UI ───────────────────────────────
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .entity-per {color: #FF4B4B; font-weight: bold;}
    .entity-org {color: #00C853; font-weight: bold;}
    .entity-loc {color: #2962FF; font-weight: bold;}
    .entity-misc {color: #FF9800; font-weight: bold;}
    .stButton>button {width: 100%;}
    </style>
""", unsafe_allow_html=True)

# ── Title & Description ────────────────────────────────────
st.title("🏷️ Named Entity Recognition (NER)")
st.markdown("**BERT-based NER Model** trained on CoNLL-2003 dataset")
st.caption("Extract **Persons, Organizations, Locations, and Miscellaneous** entities from text.")

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    - **Model**: BERT-base-cased fine-tuned on CoNLL-2003  
    - **Entities**: PER, ORG, LOC, MISC  
    - **Test F1 Score**: 0.909  
    """)
    
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Enter or paste text  
    2. Click **Extract Entities**  
    3. View highlighted results  
    """)

# ── Load Model (Cached) ───────────────────────────────────
@st.cache_resource
def load_ner_model():
    model_path = Path("best_model")  # Change this if your model is saved elsewhere
    if not model_path.exists():
        st.error("❌ Model folder 'best_model' not found. Please train and save the model first.")
        st.stop()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_ner_model()

# ── Helper Functions ─────────────────────────────────────
def predict_ner(text: str):
    tokens = text.split()
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding='max_length'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)[0]
    
    # Align predictions with original words
    word_ids = encoding.word_ids()
    predicted_labels = []
    previous_word_idx = None
    
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            label = id2label[predictions[idx].item()]
            predicted_labels.append(label)
        previous_word_idx = word_idx
    
    return tokens, predicted_labels

# Load label mapping
id2label = model.config.id2label

# ── Main Input Area ───────────────────────────────────────
st.subheader("📝 Input Text")

input_text = st.text_area(
    "Enter a sentence or paragraph:",
    height=150,
    placeholder="Elon Musk founded SpaceX and Tesla in California...",
    help="You can paste multiple sentences."
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔍 Extract Entities", use_container_width=True, type="primary"):
        if input_text.strip():
            with st.spinner("Analyzing text..."):
                tokens, labels = predict_ner(input_text.strip())
                
                # Store in session state for display
                st.session_state.tokens = tokens
                st.session_state.labels = labels
        else:
            st.warning("Please enter some text.")

with col2:
    if st.button("🧹 Clear", use_container_width=True):
        st.session_state.pop("tokens", None)
        st.session_state.pop("labels", None)
        input_text = ""

# ── Results Section ───────────────────────────────────────
if "tokens" in st.session_state and "labels" in st.session_state:
    tokens = st.session_state.tokens
    labels = st.session_state.labels
    
    st.markdown("### 📊 NER Results")
    
    # Colored Output
    st.markdown("**Highlighted Entities:**")
    colored_text = []
    for token, label in zip(tokens, labels):
        if label == 'O':
            colored_text.append(token)
        else:
            entity_type = label.split('-')[-1]
            colored_text.append(f"<span class='entity-{entity_type.lower()}'>{token}</span>")
    
    st.markdown(" ".join(colored_text), unsafe_allow_html=True)
    
    # Structured Table
    st.markdown("**Extracted Entities:**")
    entities = []
    current_entity = None
    current_tokens = []
    
    for token, label in zip(tokens, labels):
        if label.startswith('B-'):
            if current_entity:
                entities.append({
                    "Entity": " ".join(current_tokens),
                    "Type": current_entity
                })
            current_entity = label[2:]
            current_tokens = [token]
        elif label.startswith('I-') and current_entity == label[2:]:
            current_tokens.append(token)
        else:
            if current_entity:
                entities.append({
                    "Entity": " ".join(current_tokens),
                    "Type": current_entity
                })
                current_entity = None
                current_tokens = []
    
    if current_entity:
        entities.append({"Entity": " ".join(current_tokens), "Type": current_entity})
    
    if entities:
        df_entities = pd.DataFrame(entities)
        st.dataframe(df_entities, use_container_width=True)
    else:
        st.info("No named entities found in the text.")

    # Raw Output (for debugging)
    with st.expander("View Raw Token-Level Output"):
        raw_df = pd.DataFrame({"Token": tokens, "Predicted Label": labels})
        st.dataframe(raw_df, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Built for NER Practice** | Fine-tuned BERT Model")
st.caption("Note: This is an educational demonstration. Model performance may vary on domain-specific text.")