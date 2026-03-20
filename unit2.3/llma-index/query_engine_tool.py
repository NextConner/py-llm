from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# 声明向量模型
embedding_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

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
query_engine = index.as_query_engine(llm=llm)
# 通过queryEngine 转换出 queryEngineTool 
tool = QueryEngineTool.from_defaults(query_engine,name="simple query engine" ,description="good query engine for test and so on" )
print(tool)