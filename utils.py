from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, Tool, tool, VisitWebpageTool
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.base.base_query_engine import BaseQueryEngine

# 返回默认codeAgent
def get_agent() -> CodeAgent:

    return CodeAgent(
        tools=[
            DuckDuckGoSearchTool(), 
            VisitWebpageTool(),
            FinalAnswerTool()
        ], 
        model=InferenceClientModel(),
        max_steps=10,
        verbosity_level=2
    )

# 返回默认向量模型
def get_embedding_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

# 定义简单查询引擎
def get_query_engine() -> BaseQueryEngine:
     # 声明向量模型
    embedding_model = get_embedding_model()

    db = chromadb.PersistentClient(path="./alfred_chroma_db")
    # 初始化存储集
    chroma_collection = db.get_or_create_collection("alfred")
    # 初始化向量存储对象
    verctor_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # 
    index = VectorStoreIndex.from_vector_store(verctor_store,embed_model= embedding_model)
    # 指定模型
    llm = HuggingFaceInferenceAPI(model_name = "Qwen/Qwen2.5-Coder-32B-Instruct")
    #
    return index.as_query_engine(llm=llm)

# 获取 query_engine_tool 
def get_query_engine_tool() -> QueryEngineTool:
   
    query_engine = get_query_engine()
    # 通过queryEngine 转换出 queryEngineTool 
    tool = QueryEngineTool.from_defaults(query_engine,name="simple query engine" ,description="good query engine for test and so on" )
    print(tool)
    return tool