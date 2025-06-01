import os
from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import requests

app = Flask(__name__)

DB_FAISS_PATH = "vectorstore/db_faiss"
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Embedding model for FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS vectorstore
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

def call_llm(prompt: str):
    API_URL = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_REPO_ID}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    # result format example: [{"generated_text": "..."}]
    return result[0]['generated_text'] if result else "No response from LLM"

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

        # Call LLM with the full prompt text via HuggingFace Inference API
        response = call_llm(prompt_text)

        # Save current Q&A to history
        chat_history.append((query, response))

        return jsonify({
            "response": response,
            "chat_history": chat_history[-5:]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "✅ PDF Chatbot API with Chat History is running!"

if __name__ == "__main__":
    app.run(debug=True)
