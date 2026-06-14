import streamlit as st
import pandas as pd
import torch
from transformers import pipeline, set_seed

st.set_page_config(
    page_title="GPT-2 - Text Generation",
    layout="centered"
)

st.title("✍️ GPT (Generative Pre-trained Transformer)")
st.write("Generate text using pretrained GPT-2")

example_prompts = [
    "Artificial intelligence is",
    "Once upon a time in a small village,",
    "The future of space exploration"
]


@st.cache_resource
def load_model():
    return pipeline(
        task="text-generation",
        model="gpt2",
        framework="pt"
    )


with st.spinner("Loading GPT-2..."):
    generator = load_model()

st.success("Model loaded")


st.header("📊 Dataset Preview")

st.dataframe(
    pd.DataFrame(
        {"Example Prompts": example_prompts}
    ),
    use_container_width=True
)

old_prompt = st.selectbox(
    "Example Prompt",
    example_prompts
)

new_prompt = st.text_area(
    "Your Prompt",
    "The best way to learn deep learning is"
)

max_length = st.slider(
    "Output Length",
    20,
    200,
    50
)

if st.button("Generate"):

    set_seed(42)

    old_result = generator(
        old_prompt,
        max_new_tokens=max_length,
        do_sample=True
    )

    new_result = generator(
        new_prompt,
        max_new_tokens=max_length,
        do_sample=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Example")
        st.write(old_result[0]["generated_text"])

    with c2:
        st.subheader("Your Prompt")
        st.write(new_result[0]["generated_text"])