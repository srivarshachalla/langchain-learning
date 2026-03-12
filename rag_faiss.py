from dotenv import load_dotenv
load_dotenv()       

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# STEP 1 — LOAD the document

print("Step1:Loading document...")
loader = TextLoader("hr_policy.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document.")
print(f"Total characters in document: {len(documents[0].page_content)}")

# STEP 2 — SPLIT into chunks

print("\nStep2:Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")
print(f"Sample chunk:\n '{chunks[2].page_content}'")

# STEP 3 — EMBED and STORE in FAISS

print("\nStep3:Embedding and storing in FAISS...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
print("FAISS vectorstore created.")
vector_store.save_local("faiss_index")
print("saved to FAISS index/folder.")

# STEP 4 — RETRIEVE relevant chunks

print("\nStep4:Testing Retrieval...")
retriever = vector_store.as_retriever(
    search_type="similarity",  
    search_kwargs={"k": 3}  # retrieve top 3 relevant chunks
)
test_query = "How many leaves do employees get?"
retrieved = retriever.invoke(test_query)
print(f"  Query: '{test_query}'")
print(f"Retreived {len(retrieved)} chunks:")
for i, doc in enumerate(retrieved, 1):
    print(f"  chunk{i}: {doc.page_content[:80]}...")

# STEP 5 — GENERATE answer using retrieved chunks

print("\nStep5:Buiulding RAG chain...")
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an HR assistant at BigStep Technologies.
Answer questions using ONLY the context provided below.
If the answer is not in the context, say 'I don't have that information in the HR policy.'

Context:
{context}"""),
    ("human", "{question}")
])

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = ({ "context":  retriever | format_docs,
        "question": RunnablePassthrough() } | prompt | llm | parser)

print("\n" + "="*50)
print(" ASKING QUESTIONS FROM HR POLICY")
print("="*50)

questions = [
    "How many leaves do employees get per year?",
    "What is the WFH policy?",
    "How many days notice do I need to resign?",
    "What are the office hours?",
    "Can I get reimbursed for food during client visits?",
    "What is the salary of the CEO?",   # not in document — should say so!
]

for q in questions:
    print(f"\n {q}")
    answer = rag_chain.invoke(q)
    print(f" {answer}")

