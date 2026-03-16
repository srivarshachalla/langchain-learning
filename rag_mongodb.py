from dotenv import load_dotenv
load_dotenv()

import os
from pymongo import MongoClient
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Step 1: Connect 
print("Step1: Connecting to MongoDB Atlas...")
MONGODB_URI = os.getenv("MONGODB_URI")
client     = MongoClient(MONGODB_URI)
db         = client["bigstep_hr"]
collection = db["policies"]
print("   Connected ")

# Step 2: Load and split 
print("\nStep2: Loading and splitting document...")
loader   = TextLoader("hr_policy.txt")
docs     = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)
chunks = splitter.split_documents(docs)
print(f"   Created {len(chunks)} chunks ")

# Step 3: Embed and store 
print("\nStep3: Embedding and storing...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Only re-embed if collection is empty
if collection.count_documents({}) == 0:
    print("   Inserting documents...")
    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=collection,
        index_name="vector_index",
    )
    print("   Stored ")
else:
    print(f"   Already have {collection.count_documents({})} docs, skipping insert")

# Step 4: Connect vector store 
print("\nStep4: Connecting to vector store...")
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name="vector_index",
    text_key="text",
    embedding_key="embedding",
)

# DEBUG: Test with direct similarity search 
print("\nDEBUG: Direct similarity search test...")
test_query = "how many leaves do employees get"
results = vector_store.similarity_search(test_query, k=3)
print(f"   Found {len(results)} results")
if results:
    for i, r in enumerate(results, 1):
        print(f"   Result {i}: {r.page_content[:80]}...")
else:
    print(" Still no results")

#  Step 5: Build RAG chain 
print("\nStep5: Building RAG chain...")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an HR assistant at BigStep Technologies.
Answer questions using ONLY the context provided below.
If the answer is not in the context say:
'I dont have that information in the HR policy.'

Context:
{context}"""),
    ("human", "{question}")
])

llm    = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context":  retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

# Test Questions 
print("\n" + "="*50)
print("ASKING QUESTIONS FROM MONGODB ATLAS")
print("="*50)

questions = [
    "How many leaves do employees get per year?",
    "What is the WFH policy?",
    "How many days notice do I need to resign?",
    "What is the salary of the CEO?",
]

for q in questions:
    print(f"\n{q}")
    answer = rag_chain.invoke(q)
    print(f" {answer}")