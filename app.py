import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ASTUCE SÉCURISÉE : Si python-dotenv n'est pas installé, on lit le fichier .env manuellement
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("GROQ_API_KEY"):
                os.environ["GROQ_API_KEY"] = line.strip().split("=")[1].strip('"').strip("'")

# Configuration de la page Streamlit
st.set_page_config(page_title="Tunisia Startup Act - RAG Agent", page_icon="🤖", layout="wide")
st.title("🤖 Assistant Juridique Intelligent - Startup Act Tunisie")
st.write("Téléversez un document officiel (PDF) et posez vos questions à l'Agent IA.")

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.warning("⚠️ Clé GROQ_API_KEY manquante. Vérifiez que votre fichier .env contient bien votre clé sous la forme : GROQ_API_KEY=votre_clé")
    st.stop()

# Initialisation du LLM Groq (Llama 3.3 70B)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, groq_api_key=groq_api_key)

prompt_template = ChatPromptTemplate.from_template(
    "Tu es un assistant juridique expert en écosystème des startups en Tunisie.\n"
    "Réponds à la question de l'utilisateur en te basant UNIQUEMENT sur le contexte fourni ci-dessous.\n"
    "Si tu ne trouves pas la réponse dans le contexte, dis poliment que l'information ne figure pas dans le document.\n"
    "Fais des réponses claires, structurées et professionnelles en français.\n\n"
    "Contexte :\n{context}\n\n"
    "Question : {question}"
)

st.sidebar.header("📁 Document Source")
uploaded_file = st.sidebar.file_uploader("Téléversez le guide du Startup Act (PDF)", type="pdf")

@st.cache_resource
def initialiser_base_vectorielle(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    return db.as_retriever(search_kwargs={"k": 3})

retriever = None

if uploaded_file is not None:
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.get_buffer())
    
    with st.sidebar:
        with st.spinner("🧠 Analyse et indexation du PDF en cours..."):
            retriever = initialiser_base_vectorielle(temp_file_path)
        st.success("✅ Document prêt et indexé !")
else:
    st.info("💡 Veuillez téléverser un fichier PDF dans la barre latérale pour activer l'agent.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question_utilisateur := st.chat_input("Posez votre question sur le Startup Act..."):
    with st.chat_message("user"):
        st.markdown(question_utilisateur)
    st.session_state.messages.append({"role": "user", "content": question_utilisateur})
    
    with st.chat_message("assistant"):
        if retriever is None:
            reponse_texte = "Désolé, je ne peux pas répondre tant que vous n'avez pas téléversé de document PDF."
            st.markdown(reponse_texte)
        else:
            with st.spinner("⏳ Recherche dans le document..."):
                try:
                    documents_trouves = retriever.invoke(question_utilisateur)
                    contexte_fusionne = "\n\n".join([doc.page_content for doc in documents_trouves])
                    prompt_final = prompt_template.format(context=contexte_fusionne, question=question_utilisateur)
                    reponse = llm.invoke(prompt_final)
                    reponse_texte = reponse.content
                    st.markdown(reponse_texte)
                except Exception as e:
                    reponse_texte = f"❌ Une erreur est survenue lors de la communication avec Groq : {e}"
                    st.error(reponse_texte)
                    
        st.session_state.messages.append({"role": "assistant", "content": reponse_texte})