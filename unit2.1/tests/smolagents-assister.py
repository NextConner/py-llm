import os
import json
from typing import Optional
from smolagents import CodeAgent, InferenceClientModel, tool

#模型定义
model = InferenceClientModel(
    "Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=8096
)

#工具定义
@tool
def get_weather(city:str , country:Optional[str] = "CN") -> str:
    """
    GET the current weather for a given city
    Args:
        city: The name of the city (e.g., "上海").
        country: The country code (default is "CN").
    """

    #返回模拟数据
    import random 
    temp = random.randint(15,35)
    conditions = ["Sunny","Cloudy","Rainy"]
    condition = random.choice(conditions)
    return f"Weather in {city}, {country}: {temp}·C ， {condition}"

# todo tool

TODO_FILE = "todo.json"

@tool
def add_todo(task:str) -> str:
    """
    Add a new taks to the todo list.
    Args:
        task: The description of the task.
    """
    # 读取存在待办
    if os.path.exists(TODO_FILE):
        with open(TODO_FILE,'r') as f:
            todos = json.load(f)
    else:
        todos = []

    
    # 添加
    new_id = len(todos) + 1
    todos.append({"id":new_id,"task":task,"done":False})

    # 写回文件
    with open(TODO_FILE,'w') as f:
        json.dump(todos,f,indent=2)
    
    return f"Task '{task}' added with ID {new_id}."

# 查看todo 工具
@tool
def list_todos() -> str:
    """List all current todo item"""
    if not os.path.exists(TODO_FILE):
        return "No todos found,list is empty"

    with open(TODO_FILE,"r") as f:
        todos = json.load(f)
    
    if not todos:
        return "The todo list is empty."

    result = "Current Todos:\n"
    for item in todos:
        status = "✓" if item["done"] else "☐"
        result += f"{item['id']}. {status} {item['task']}\n"
    return result

# 工具4：标记完成
@tool
def complete_todo(todo_id: int) -> str:
    """
    Mark a todo item as completed.
    Args:
        todo_id: The ID of the task to mark as done.
    """
    if not os.path.exists(TODO_FILE):
        return f"Error: No todo list found. Cannot complete ID {todo_id}."
    
    with open(TODO_FILE, 'r') as f:
        todos = json.load(f)
    
    for item in todos:
        if item["id"] == todo_id:
            item["done"] = True
            with open(TODO_FILE, 'w') as f:
                json.dump(todos, f, indent=2)
            return f"Marked task '{item['task']}' as completed."
    
    return f"Error: Todo item with ID {todo_id} not found."


# 构建 agent 
agent  = CodeAgent(
    tools=[add_todo,list_todos,get_weather,complete_todo],
    model = model
)

# 
if __name__ == "__main__":
    print("🤖 Personal Assistant Agent is ready!")
    print("Try: 'What's the weather in Shanghai?' or 'Add a task to buy milk'")
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower in ['quit','exit','q']:
                break 
            result = agent.run(user_input)
            print(f"Assistant:{result}")
        except KeyboardInterrupt:
            print("\nGoodbye")
            break