from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

template = ChatPromptTemplate.from_messages([
    ("system",  """You are an AI development assistant at BigStep Technologies.
You help developers understand Python, LangChain, LangGraph, 
n8n, and AI tools clearly.
Always give accurate technical explanations.
Keep answers to 3-4 sentences maximum."""),
    ("human", "Explain {topic} to {name} in simple terms.")])

llm = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

    # ── Connect them with pipe ─────────────────────────────
chain = template | llm | parser

print("---Run 1------")
result1 = chain.invoke({"topic": "vector databases", "name" : "varsha"}) 
print(result1)
print()

print("---Run 2------")
result2 = chain.invoke({"topic": "langchain", "name": "varsha"})
print(result2)
print()

print("---Run 3------")
result3 = chain.invoke({"topic": "langgraph", "name": "varsha"})
print(result3)
