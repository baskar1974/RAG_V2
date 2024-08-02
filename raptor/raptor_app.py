import gradio as gr
from raptor_demo import answer_generator

gr.ChatInterface(answer_generator).launch(share=True)
