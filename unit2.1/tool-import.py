from smolagents import load_tool, CodeAgent, InferenceClientModel
import os
from huggingface_hub import login

login(os.getenv("HF_TOKEN"))
image_generation_tool = load_tool(
    "m-ric/text-to-image",
    trust_remote_code=True
)

agent = CodeAgent(
    tools=[image_generation_tool],
    model=InferenceClientModel()
)

agent.run("Generate an image of the cat of tom and jerry , they eating checking together.")