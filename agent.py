from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def get_leave_balance(employee_name: str) -> str:
    """Gets the remaining leave balance for the an employee.
    Use this when someone asks about leave balance or remaining leaves."""
    # Simulated data for demonstration
    balances  = {
        "varsha": "12",
        "rahul": "8",
        "priya": "15"
    }
    name = employee_name.lower()
    if name in balances:
        return f"{employee_name} has {balances[name]} leaves remaining."
    return f"Employee {employee_name} not found in the system"

@tool
def get_company_policy(policy_name: str) -> str:
    """Gets BigStep Technologies company policy on a given topic.
    Use this when someone asks about company rules, policies, or guidelines."""
    # Simulated data for demonstration
    policies = {
        "leave":     "Employees get 24 leaves per year. Unused leaves can be carried forward up to 10 days.",
        "wfh":       "Employees can work from home up to 2 days per week with manager approval.",
        "appraisal": "Performance appraisals happen every 6 months in April and October.",
        "timing":    "Office hours are 9:30 AM to 6:30 PM Monday to Friday.",
    }
    topic_lower = policy_name.lower()
    for key, value in policies.items():
        if key in topic_lower:
            return value
    return f"No policy found for {topic}. Please contact HR directly."

@tool
def calculate_working_hours(start_date: str, end_date: str) -> str:
    """Calculates the number of working days between two dates.
    Use this when someone wants to know how many working days are between dates.
    Dates should be in YYYY-MM-DD format."""
    from datetime import datetime, timedelta
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        count = 0
        current = start
        while current <= end:
            if current.weekday() < 5:   # Monday=0, Friday=4
                count += 1
            current += timedelta(days=1)
        return f"There are {count} working days between {start_date} and {end_date}."
    except ValueError:
        return "Invalid date format. Please use YYYY-MM-DD format."
    
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_leave_balance, get_company_policy, calculate_working_hours]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="You are a helpful HR assistant at BigStep Technologies. "
           "Use the available tools to answer employee questions accurately."
) 

from langchain_core.messages import HumanMessage

def ask_agent(question):
   print(f"You: {question}")
   result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })
   
   print(f"Bot: {result['messages'][-1].content}")
   print()


ask_agent("What is Varsha's leave balance?")
ask_agent("What is the WFH policy at BigStep?")
ask_agent("How many working days are between 2025-06-01 and 2025-06-30?")
ask_agent("What is Rahul's leave balance and what is the leave policy?")