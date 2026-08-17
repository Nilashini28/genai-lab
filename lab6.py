from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# Documents
documents = [
    "Artificial Intelligence is the simulation of human intelligence by machines.",
    "Machine Learning is a subset of Artificial Intelligence.",
    "Deep Learning uses neural networks with multiple layers.",
    "Natural Language Processing enables computers to understand human language."
]

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector store
vectorstore = FAISS.from_texts(
    documents,
    embedding=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# User query
query = "What is Machine Learning?"

# Retrieve relevant documents
docs = retriever.invoke(query)

context = "\n".join(
    [doc.page_content for doc in docs]
)

# Text generation model
generator = pipeline(
    "text-generation",
    model="gpt2"
)

prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

output = generator(
    prompt,
    max_length=150,
    num_return_sequences=1,
    do_sample=False
)

print(output[0]["generated_text"])