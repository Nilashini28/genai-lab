import gradio as gr
from transformers import pipeline

# Load text-generation model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# Inference function
def generate_text(prompt):
    result = generator(
        prompt,
        max_length=100
    )
    return result[0]["generated_text"]


# Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(
        label="Enter your prompt"
    ),
    outputs=gr.Textbox(
        label="Generated Text"
    ),
    title="Generative AI Text Generator"
)

# Launch application
demo.launch()