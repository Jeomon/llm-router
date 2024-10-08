from src.router.utils import read_markdown_file
from src.inference import BaseInference
from src.message import HumanMessage,SystemMessage
from json import dumps

class LLMRouter:
    def __init__(self,routes:list[dict]=[],llm:BaseInference=None):
        self.system_prompt=read_markdown_file('./src/router/prompt.md')
        self.routes=dumps(routes,indent=2)
        self.llm=llm

    def invoke(self,query:str)->dict:
        messages=[SystemMessage(self.system_prompt.format(routes=self.routes)),HumanMessage(query)]
        response=self.llm.invoke(messages,json=True)
        route=response.content
        return route
