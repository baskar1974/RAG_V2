import torch
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig, RetrievalAugmentation
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
from langchain_community.document_loaders import PyPDFLoader
import os
import torch

# You can define your own Summarization model by extending the base Summarization Class. 
class SummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="intfloat/e5-base-v2"):
        # Initialize the tokenizer and the pipeline for the GEMMA model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarization_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device('cuda'),  # Use "cpu" if CUDA is not available
        )

    def summarize(self, context, max_tokens=150):
        # Format the prompt for summarization
        messages=[
            {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate the summary using the pipeline
        outputs = self.summarization_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        # Extracting and returning the generated summary
        summary = outputs[0]["generated_text"].strip()
        return summary
    
class QAModel(BaseQAModel):
    def __init__(self, model_name= "meta-llama/Meta-Llama-3-8B-Instruct"):
        # Initialize the tokenizer and the pipeline for the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qa_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device('cuda'),
        )

    def answer_question(self, context, question):
        # Apply the chat template for the context and question
        messages=[
              {"role": "user", "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate the answer using the pipeline
        outputs = self.qa_pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        # Extracting and returning the generated answer
        answer = outputs[0]["generated_text"][len(prompt):]
        return answer
    

class EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)

RAC = RetrievalAugmentationConfig(summarization_model=SummarizationModel(), qa_model=QAModel(), embedding_model=EmbeddingModel())

RA = RetrievalAugmentation(config=RAC)

pdf_folder_path = "/home/bmuthiah/LLM/RAG_V2/raptor/demo/calix_pdf/ragdataset"
documents = []
for file in os.listdir(pdf_folder_path):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
cleaned_doc = [x for x in documents if x != 'NaN']

RA.add_documents(str(cleaned_doc))

question = "Summary of high level BSP calix?"

answer = RA.answer_question(question=question)

print("Answer: ", answer)

#done

