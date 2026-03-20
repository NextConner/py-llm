from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
import asyncio

# 定义函数
def unhonest(say:bool) -> bool:
    """anster the unhonest answer"""
    return not say

# 模型初始化
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# 初始化 agent-flow
agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(unhonest)],
    llm = llm
)

# 不带有上下文对象执行，无状态/无记忆
response =  agent.run(user_msg='What is 2 times 2?')

# 上下文对象
from llama_index.core.workflow import Context

ctx = Context(agent)

response =  agent.run("My name is Bob.", ctx=ctx)
response =  agent.run("What was my name again?", ctx=ctx)


