import os
from datetime import datetime
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent # Requires langchain >= 0.3.0
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# 1. Load environment variables
load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")

# 2. Define the tool with a decorator for proper metadata
@tool
def get_date():
    """Returns the current date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")

# 3. Initialize the Gemini model
# Gemini 3.0+ models require a temperature of 1.0 or higher
llm = ChatOllama(
    model="qwen3.5:4b", 
    # google_api_key=api_key,
    temperature=1.0 
)

# 4. Create the agent
system_prompt = "You are a helpful assistant."
agent = create_agent(model=llm, tools=[get_date], system_prompt=system_prompt)

# 5. Execute the agent (not the llm directly)
user_query = input("Enter a query (e.g., 'What is today's date?'): ")
response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})

# 6. Display the final response
print(response['messages'][-1].content)
