import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Load models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

llm = HuggingFaceEndpoint(
    repo_id=HUGGINGFACE_REPO_ID,
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.5,
    max_new_tokens=512
)

prompt_template = PromptTemplate(
    template="""
You are a helpful assistant. Use the conversation history and context to answer the user's question.
If you don’t know the answer, just say you don’t know. Don’t try to make up an answer.
Don’t provide anything out of the given context.

Conversation history:
{history}

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
""",
    input_variables=["history", "context", "question"]
)

chat_history = []

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    history_text = "\n".join(
        [f"User: {q}\nAssistant: {a}" for q, a in chat_history[-5:]]
    ) if chat_history else "No previous conversation."

    docs = vectorstore.as_retriever(search_kwargs={'k': 3}).get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    full_prompt = prompt_template.format(
        history=history_text,
        context=context,
        question=query
    )

    response = llm(full_prompt)
    chat_history.append((query, response))

    return jsonify({
        "response": response,
        "chat_history": chat_history[-5:]
    })

@app.route("/", methods=["GET"])
def root():
    return "✅ PDF Chatbot API with Chat History is running!"

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7860)
