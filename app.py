import streamlit as st
import pandas as pd
from transformers import pipeline, set_seed

st.set_page_config(page_title="GPT-2 - Text Generation", layout="centered")
st.title("✍️ GPT (Generative Pre-trained Transformer)")
st.markdown("Generate text using a pretrained GPT-2 model from Hugging Face.")

example_prompts = [
    "Artificial intelligence is",
    "Once upon a time in a small village,",
    "The future of space exploration",
]

# ----------------------------------------------------------------------
# 2. Model Loading (GPT-2 is pretrained, so no training step is needed)
# ----------------------------------------------------------------------
st.header("🏋️ Model Loading")


@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")


with st.spinner("Loading pretrained GPT-2 model..."):
    generator = load_model()

st.success("Pretrained GPT-2 model loaded successfully! (No training required.)")

# ----------------------------------------------------------------------
# 1. Dataset Preview
# ----------------------------------------------------------------------
st.header("📊 Dataset Preview")
st.markdown("A few example prompts you can try with GPT-2:")
st.dataframe(pd.DataFrame({"Example Prompts": example_prompts}), use_container_width=True)

# ----------------------------------------------------------------------
# 3. Output - Old Data vs New Data Comparison
# ----------------------------------------------------------------------
st.header("🔮 Output")
st.markdown("Compare GPT-2's generated text for an **example prompt** (old data) with text generated for **your own prompt** (new data).")

old_prompt = st.selectbox("Pick an example prompt:", example_prompts, index=0)
new_prompt = st.text_area("Enter your own new prompt:", "The best way to learn deep learning is")
max_length = st.slider("Max length of generated text", 20, 200, 50)

if st.button("Compare Generations"):
    set_seed(42)
    old_result = generator(old_prompt, max_length=max_length, num_return_sequences=1, truncation=True)
    new_result = generator(new_prompt, max_length=max_length, num_return_sequences=1, truncation=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 Old Data (Example Prompt)")
        st.write(f"**Prompt:** {old_prompt}")
        st.write("**Generated Text:**")
        st.write(old_result[0]["generated_text"])
    with col2:
        st.subheader("🆕 New Data (Your Prompt)")
        st.write(f"**Prompt:** {new_prompt}")
        st.write("**Generated Text:**")
        st.write(new_result[0]["generated_text"])