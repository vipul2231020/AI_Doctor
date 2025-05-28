import os
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

# Setup HuggingFace Token and Model ID

HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Load LLM from HuggingFace
def load_llm(repo_id):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        max_new_tokens=512,  # replace max_length with max_new_tokens
        huggingfacehub_api_token=HF_TOKEN
    )


# Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don’t know the answer, just say you don’t know. Don’t try to make up an answer.
Don’t provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Build the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Query loop (one-time or repeatable)
while True:
    user_query = input("\n💬 Write Query Here (or type 'exit' to quit): ")
    if user_query.lower() in ['exit', 'quit']:
        break

    try:
        response = qa_chain.invoke({'query': user_query})  # ✅ Correct key here
        print("\n📌 RESULT:\n", response["result"])

        # print("\n📚 SOURCE DOCUMENTS:")
        # for i, doc in enumerate(response["source_documents"], 1):
        #     print(f"\n--- Source #{i} ---")
        #     print(doc.page_content[:500])  # Show a snippet of the content

    except Exception as e:
        print("❌ Error:", e)
