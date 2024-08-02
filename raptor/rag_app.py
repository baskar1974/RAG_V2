import gradio as gr
from RAG.AnswerGenerator import answer_generator

gr.ChatInterface(answer_generator).launch(share=True)
