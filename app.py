from src.router import LLMRouter
from src.inference.groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()
api_key=os.environ.get("GROQ_API_KEY")

routes=[
    {
        'route':'code',
        'description':'route to handle queries about code generation or writing code'
    },
    {
        'route':'debug',
        'description':'route to handle queries about debugging the code or fixing the error in the code'
    },
    {
        'route':'misc',
        'description':'route to handle miscellaneous queries such as code review, documentation,..etc'
    }
]

llm=ChatGroq(model='llama3-70b-8192',api_key=api_key,temperature=0)
router=LLMRouter(routes=routes,llm=llm)
print(router.invoke("Create a program to count the number of lines in a text file"))