from llama_index.llms.deepseek import DeepSeek
import os
from dotenv import load_dotenv


# 记载.env
load_dotenv()


# 获取配置的hf token 

ds_token = os.getenv("DESPSSEK_API_TOKEN")

llm = DeepSeek(model="deepseek-chat", api_key=ds_token)


from llama_index.core import SimpleDirectoryReader
# llm 加载本地文档
reader = SimpleDirectoryReader(input_dir="./pdfs")
documents = reader.load_data()

from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
)

nodes =  pipeline.arun(documents=[Document.example()])