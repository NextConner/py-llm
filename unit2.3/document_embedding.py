from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
import asyncio

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
)

async def main():
    # ✅ 正确：在 async def 函数内部使用 await
    nodes = await pipeline.arun(documents=[Document.example()])
    # 对 nodes 进行后续处理...
    return nodes

# 启动并运行异步主函数
if __name__ == "__main__":
    nodes = asyncio.run(main())
    print(nodes)