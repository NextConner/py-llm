from smolagents import CodeAgent, DuckDuckGoSearchTool, FinalAnswerTool, InferenceClientModel, Tool, tool, VisitWebpageTool

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