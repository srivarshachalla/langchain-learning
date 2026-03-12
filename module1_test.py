from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

# Test 1 -- LangChain real API call
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke([HumanMessage(content="Say hello in one sentence.")])
print("✅ LangChain working:", response.content)

# Test 2 -- LangGraph import
graph = StateGraph(dict)
print("✅ LangGraph working!")