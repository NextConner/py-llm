from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, InferenceClientModel

# 基于tool类定义工具
class PartyPlanningRetrieverTool(Tool):
    #工具名称
    name = "PartyPlanningRetrieverTool"
    #工具描述
    description = "A tool for retrieving information about party planning from a document store."
    #输入描述
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to retrieve information about party planning."
        }
    }
    #输出描述
    output_type: "string"
    outputs = {
        "output_type": "string",  # 键也可能是 "type" 而不是 "output_type"
        "description": "The retrieved information about party planning."
    }

    #初始
    def __init__(self, documents,**kwargs):
       super().__init__(**kwargs)
       #检索文档，并设置k=5，返回前5条相关文档
       self.retriever = BM25Retriever.from_documents(documents,k=5)
    
    #执行逻辑函数
    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Input query must be a string."

        docs = self.retriever.invoke(query,)

        return "\nRetrieved docus:\n" + "".join(
            [
                f"\n\n====Idea {str(i)} =====\n" + docs.page_content
                for i,doc in enumerate(docs)
            ]
        )
    

# 模拟数据
party_documens = [
    {"text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.", "source": "Party Ideas 1"},
    {"text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.", "source": "Entertainment Ideas"},
    {"text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'", "source": "Catering Ideas"},
    {"text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.", "source": "Decoration Ideas"},
    {"text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.", "source": "Entertainment Ideas"}
]

source_docs = [
    Document(page_content=doc["text"],method={"source":doc["source"]})
    for doc in party_documens
]

# 文档分块化
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index = True,
    strip_whitespace = True,
    separators=["\n\n","\n",".",","," ",""],
)

#模拟数据向量化
docs_processed = text_splitter.split_documents(source_docs)

#检索tool
plan_retriever = PartyPlanningRetrieverTool(docs_processed)

#初始化agent
agent = CodeAgent(tools=[plan_retriever],model=InferenceClientModel())

#使用
response = agent.run(
    "Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options."
)

print(response)