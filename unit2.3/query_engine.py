from llama_index.llms.deepseek import DeepSeek
from llama_index.core import VectorStoreIndex,Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.embeddings.alibabacloud_aisearch import (
    AlibabaCloudAISearchEmbedding,
)

import os
from dotenv import load_dotenv


# 记载.env
load_dotenv()

llm = AlibabaCloudAISearchEmbedding()

index = VectorStoreIndex()

query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)

response = query_engine.query("quick answer 1 + 1 = ? ")
print(response)