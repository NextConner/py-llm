from llama_index.tools.mcp import BasicMCPClient,McpToolSpec
from utils import get_agent

mcp_client = BasicMCPClient("http://127.0.0.1:8080/sse")
# 转换mcp client 成 mcp tool 
mcp_tool = McpToolSpec(client=mcp_client)

agent = get_agent(mcp_tool)

agent_context = Context(agent)