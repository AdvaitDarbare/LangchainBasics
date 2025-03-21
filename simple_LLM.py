from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
print("LangSmith Project:", os.getenv("LANGSMITH_PROJECT"))

# Setup LangChain LLM
llm = OllamaLLM(model="llama3.2")

# Create a prompt
prompt = PromptTemplate.from_template("What's a fun fact about {topic}?")

# Create a chain (updated version)
chain = RunnableSequence(prompt | llm)

# Run it (tracked in LangSmith)
response = chain.invoke({"topic": "AI agents"})
print(response)
