import os
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

app = Flask(__name__)

DB_FAISS_PATH = "vectorstore/db_faiss"
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

chat_history = []

CUSTOM_PROMPT_TEMPLATE = """
You are a helpful assistant. Use the conversation history and context to answer the user's question.
If you don’t know the answer, just say you don’t know. Don’t try to make up an answer.
Don’t provide anything out of the given context.

Conversation history:
{history}

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["history", "context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512
    )
    return llm

llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        query = data.get("query")
        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Format conversation history string (last 5 exchanges)
        history_text = "\n".join(
            [f"User: {q}\nAssistant: {a}" for q, a in chat_history[-5:]]
        ) if chat_history else "No previous conversation."

        # Retrieve relevant documents for current query (context)
        docs = vectorstore.as_retriever(search_kwargs={'k': 3}).get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Format prompt text with history, context, and question
        prompt_text = prompt_template.format(
            history=history_text,
            context=context,
            question=query
        )

        # Call LLM with the full prompt text
        response = llm(prompt_text)

        # Save current Q&A to history
        chat_history.append((query, response))

        return jsonify({
            "response": response,
            "chat_history": chat_history[-5:]
            # "sources": [doc.page_content[:300] for doc in docs]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "✅ PDF Chatbot API with Chat History is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
