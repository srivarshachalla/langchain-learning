from dotenv import load_dotenv
load_dotenv()   

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


llm = ChatOpenAI(model="gpt-4o-mini")

history = [
    SystemMessage(content=""""You are a helpful AI assistant 
    at BigStep Technologies. You help with Python, LangChain, 
    and AI development questions. Be concise and friendly.""")]

print("Bigstep AI Assitant")
print("Type 'quit' to exit, 'history' to see conversation\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() == "quit":
        print("Bot : Goodbye! Happy coding!")
        break
    if user_input.lower() == "history":
        print("\n--- Conversation History ---")
        for msg in history:
            if isinstance(msg, SystemMessage):
                print(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                print(f"You: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"AI: {msg.content}")
        print("----------------------------\n")
        continue
    
  
    history.append(HumanMessage(content=user_input))

    response = llm.invoke(history)
    history.append(response)
    print(f"Bot: {response.content}\n")