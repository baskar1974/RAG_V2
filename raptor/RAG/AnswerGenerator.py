from transformers import pipeline, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import accelerate
import torch
import json
from DataEmbedding import load_chunk_persist_pdf

READER_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    READER_MODEL_NAME, quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)

prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
    },
]
RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    prompt_in_chat_format,
    #messages, 
    tokenize=False, add_generation_prompt=True
)

main_retriever = load_chunk_persist_pdf()

def answer_generator(questions, history):
    retrieved_docs = main_retriever.get_relevant_documents(questions)
    retrieved_docs_text = [
        doc.page_content for doc in retrieved_docs
    ]  # we only need the text of the documents
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)]
    )
    final_prompt = RAG_PROMPT_TEMPLATE.format(
        question=questions, context=context
    )
        
    # Redact an answer
    answer = READER_LLM(final_prompt)[0]["generated_text"]
    return answer

