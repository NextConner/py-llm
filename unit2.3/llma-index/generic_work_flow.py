from llama_index.core.workflow import StartEvent,StopEvent,Workflow,step

# 继承 WorkFlow 实现
class MyWorkFlow(Workflow):
    @step
    async def my_step(self,ev: StartEvent) -> StopEvent:
        print("do my step")
        return StopEvent(result="Hello world step")
    

w = MyWorkFlow(timeout=10,verbose= False)
result =  w.run