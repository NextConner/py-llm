from llama_index.core.tools import FunctionTool


###  llama FunctionTool 把函数转换成 tool , 类似 @tool 和 继承 Tool 效果

#定义查询函数
def get_weather(location: str) -> str:
    """Useful fro getting the weather for a given location"""
    print(f"Get weather for {location}")
    return f"Weather in {location} is sunny"

#实例化自定义tool 
tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location"
)
tool.call("New York")