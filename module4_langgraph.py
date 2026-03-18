from dotenv import load_dotenv
load_dotenv()   

import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

llm = ChatOpenAI(model="gpt-4o-mini")

# PART 1 — Hello World Graph
# The simplest possible LangGraph

print("PART 1 - Hello World Graph")

def hello_node(state: MessagesState):
    """A node that calls the LLM and returns its response."""
    response = llm.invoke(state["messages"])
    return {"messages" : [response]}

builder = StateGraph(MessagesState)

builder.add_node("hello", hello_node)   

builder.add_edge(START, "hello")
builder.add_edge("hello", END)  

graph = builder.compile()

result = graph.invoke({
    "messages": [HumanMessage(content="Say hello in one sentence.")]
})

print(f"Input: 'Say hello in one sentence.'")
print(f"Output: '{result['messages'][-1].content}'")
print(f"Total messages in state: {len(result['messages'])}")

# PART 2 — Two Node Graph (pipeline)
# Shows how state flows between nodes

print("PART 2 - Two Node Graph")

def node_translate(state: MessagesState):
    """A node that translates the last message to French."""
    messages = state["messages"]
    prompt = [
        SystemMessage(content="Translate the user message to French. Reply ONLY with the French translation."),
        HumanMessage(content=messages[-1].content)
    ]
    response = llm.invoke(prompt)
    print(f"  [node_translate] → '{response.content}'")
    return {"messages": [response]}

def node_summarize(state: MessagesState):
    """Node 2: Take the French text and summarize it in English."""
    messages = state["messages"]
    prompt = [
        SystemMessage(content="Summarize the user message in one sentence."),
        HumanMessage(content=messages[-1].content)
    ]
    response = llm.invoke(prompt)
    print(f"  [node_summarize] → '{response.content}'")
    return {"messages": [response]}

builder2 = StateGraph(MessagesState)
builder2.add_node("translate", node_translate)
builder2.add_node("summarize", node_summarize)
builder2.add_edge(START,       "translate")
builder2.add_edge("translate", "summarize")  
graph2 = builder2.compile()
result2 = graph2.invoke({
    "messages": [HumanMessage(content="LangGraph is a library for building stateful multi-agent applications.")]
})
print(f"\nFinal output: {result2['messages'][-1].content}")
print(f"Total messages in state: {len(result2['messages'])}")

# PART 3 — Conditional Edge (branching)
# The graph decides which node to go to next

print("\nPART 3 - Conditional Edge (branching)")

def node_classify(state: MessagesState):
    question = state["messages"][0].content
    prompt = [
        SystemMessage(content="Classify as 'HR' (leaves, WFH, salary, appraisal, office) or 'GENERAL'. Reply ONE word only: HR or GENERAL."),
        HumanMessage(content=question)
    ]
    label = llm.invoke(prompt).content.strip().upper()
    print(f"  [node_classify] → '{label}'")
    return {"messages": [AIMessage(content=f"CATEGORY:{label}")]}

def node_hr_answer(state: MessagesState):
    question = state["messages"][0].content
    prompt = [
        SystemMessage(content="You are an HR assistant at BigStep Technologies. Answer concisely."),
        HumanMessage(content=question)
    ]
    response = llm.invoke(prompt)
    print(f"  [node_hr_answer] → HR response generated")
    return {"messages": [response]}

def node_general_answer(state: MessagesState):
    question = state["messages"][0].content
    prompt = [
        SystemMessage(content="You are a general assistant. Answer concisely."),
        HumanMessage(content=question)
    ]
    response = llm.invoke(prompt)
    print(f"  [node_general_answer] → General response generated")
    return {"messages": [response]}

def route_question(state: MessagesState):
    last_message = state["messages"][-1].content  # ← must be inside function
    if "CATEGORY:HR" in last_message:
        return "hr_answer"
    return "general_answer"

builder3 = StateGraph(MessagesState)
builder3.add_node("classify",       node_classify)
builder3.add_node("hr_answer",      node_hr_answer)
builder3.add_node("general_answer", node_general_answer)
builder3.add_edge(START, "classify")
builder3.add_conditional_edges("classify", route_question)
builder3.add_edge("hr_answer",      END)
builder3.add_edge("general_answer", END)
graph3 = builder3.compile()

# ← Make sure this is a proper list with square brackets
test_questions = [
    "How many leaves do I get per year?",
    "What is the capital of France?",
    "What is the WFH policy?",
    "Who invented Python?",
]

for q in test_questions:
    print(f"\n Question: '{q}'")
    result3 = graph3.invoke({"messages": [HumanMessage(content=q)]})
    print(f" Answer:   {result3['messages'][-1].content[:120]}")
