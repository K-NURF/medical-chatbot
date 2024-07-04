import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from embedding import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="monotykamary/medichat-llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text

# Initialize Streamlit app.
def main():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.title("Chat with the Maternal Health Care Bot :hospital:")

    user_question = st.text_input("Enter your question:")

    if st.button("Submit"):
        if user_question:
            response, response_text = query_rag(user_question)
            st.session_state.chat_history.append(("user", user_question))
            st.session_state.chat_history.append(("bot", response))

    # Display chat history
    for sender, message in st.session_state.chat_history:
        if sender == "user":
            st.write(f"**User:** {message}")
        else:
            st.write(f"**Bot:** {message}")

if __name__ == "__main__":
    main()
