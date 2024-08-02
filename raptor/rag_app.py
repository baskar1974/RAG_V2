import gradio as gr
from RAG import AnswerGenerator

gr.ChatInterface(answer_generator).launch(share=True)
