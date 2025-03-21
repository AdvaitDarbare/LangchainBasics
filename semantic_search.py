
from dotenv import load_dotenv
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Step 1: Load environment variables (LangSmith API Key)
load_dotenv()

# Step 2: Load PDF using PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("./L16.pdf")
docs = loader.load()

# Step 3: Split text into chunk, smaller overlapping chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = splitter.split_documents(docs)

# Step 4: Create embeddings, turing the text into numbers (vectors) using hugging face model
# this is how we later can search for similar chunks
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 5: Store chunks in a vector store (Chroma)
# retriever is used to search for similar chunks based on the question we ask
# Chroma is a vector store that allows us to efficiently search for similar vectors
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(all_splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Step 6: Create Ollama LLM and Prompt
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableMap

llm = OllamaLLM(model="llama3.2")
# prompt template to format the context and question for the LLM to answer
prompt = PromptTemplate.from_template("""
Use the following context to answer the question:
{context}

Question: {question}
""")

# Step 7: Create retrieval + prompt + LLM chain
# RunnableMap is used to first retrieve the context based on the question
# and then pass both the context and question to the prompt
# and finally to the LLM for generating the answer
# The chain is a sequence of operations that processes the input data
# through each step: retrieval, prompt formatting, and LLM generation
chain = (
    RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"]
    })
    | prompt
    | llm
)

# Step 8: Ask a question
question = "What are indexes in database management systems?"
response = chain.invoke({"question": question})
print(response)
