from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
import os
from env_variables import MODEL_PATH
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Préparation de la base de connaissances
def prepare_knowledge_base(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = FAISS.from_documents(texts, embeddings)
    
    return db

def load_documents(source_type, source_path):
    if source_type == "text":
        loader = TextLoader(source_path, encoding='utf-8')
    elif source_type == "pdf":
        loader = PyPDFLoader(source_path)
    elif source_type == "web":
        loader = WebBaseLoader(source_path)
    elif source_type == "directory":
        loader = DirectoryLoader(source_path, glob="**/*.txt")  # Charge tous les fichiers .txt dans le dossier et ses sous-dossiers
    elif source_type == "docx":
        loader = UnstructuredWordDocumentLoader(source_path)
    else:
        raise ValueError(f"Type de source non supporté: {source_type}")

    documents = loader.load()
    
    # Diviser les documents en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    return texts

#check if rag_db exists and load it if not create it
if os.path.exists("examples/LLM/rag_db"):
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
    db = FAISS.load_local("examples/LLM/rag_db", embeddings, allow_dangerous_deserialization=True)
else:
    texts = load_documents(source_type = 'text', source_path = 'examples/LLM/Rag_document_example1.txt')
    db = prepare_knowledge_base(texts)
    #save the db 
    db.save_local("examples/LLM/rag_db")

# Configuration du modèle LLM
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.3,
    max_tokens=600,
    top_p=0.9, # Top-p sampling pour éviter les réponses trop redondantes
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=2048,  # Contexte pour Mistral-7B
)

# Préparation du prompt
template = """Utilisez uniquement les informations suivantes pour répondre à la question de l'utilisateur. 
Si l'information n'est pas présente dans le contexte, répondez simplement "Je n'ai pas assez d'informations pour répondre à cette question."
Contexte :
{context}
Question : {question}
Réponse (répondez directement sans répéter la question et sans ajouter d'informations qui ne sont pas dans le contexte fourni) :"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Configuration de la chaîne RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={
        "prompt": prompt,
        "verbose": True,
        "output_key": "answer"  # Spécifie la clé de sortie pour la réponse
    }
)

# Utilisation
question = "Parle moi de l'évolution de diablo 4 par rapport à diablo 3 (qu'elle classe est équivalente au sacresprit)"
result = qa_chain.invoke({"query": question})
