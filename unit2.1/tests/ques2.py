from smolagents import DuckDuckGoSearchTool,ToolCallingAgent,InferenceClientModel,CodeAgent

model = InferenceClientModel(
    "Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=8096
)

# Create web agent and manager agent structure
web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool],           # Add required tools
    model=model,         # Add model
    max_steps=10,        # Adjust steps
    name="web_agent",           # Add name
    description="the agent for web search and return  response "      # Add description
)

manager_agent = CodeAgent(
    model=InferenceClientModel("deepseek-ai/DeepSeek-R1", provider="together", max_tokens=8096),
    tools=[DuckDuckGoSearchTool],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    max_steps=15,
)