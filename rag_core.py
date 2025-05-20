# rag_core.py

import os
from dotenv import load_dotenv

# Import necessary components from LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Changed from langchain_openai to langchain_community for HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# Changed from langchain_openai.ChatOpenAI to langchain_community.llms.HuggingFaceHub
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from a .env file
load_dotenv()

# --- IMPORTANT: Choose ONE API Key ---
# Ensure ONLY ONE of these is uncommented and set.
# If using Hugging Face, ensure HUGGINGFACEHUB_API_TOKEN is set.
# If using OpenAI, ensure OPENAI_API_KEY is set.

# Check for Hugging Face API token
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    # If Hugging Face token is not found, check for OpenAI token as a fallback
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Neither HUGGINGFACEHUB_API_TOKEN nor OPENAI_API_KEY found. Please set one in your .env file.")
    else:
        print("Using OpenAI API key as HUGGINGFACEHUB_API_TOKEN is not set.")
        USE_OPENAI = True
else:
    print("Using Hugging Face Hub API token.")
    USE_OPENAI = False


def create_vector_store_from_pdfs(pdf_paths):
    """
    Loads PDF documents from the given paths, splits them into manageable chunks,
    creates numerical embeddings for each chunk, and stores these embeddings
    in a FAISS vector store for efficient similarity search.

    Args:
        pdf_paths (list): A list of file paths to the PDF documents.

    Returns:
        FAISS: An initialized FAISS vector store containing the document chunks,
               or None if no documents could be loaded or processed.
    """
    if not pdf_paths:
        print("No PDF paths provided. Returning None.")
        return None

    all_docs = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(pdf_path)
            all_docs.extend(loader.load())
            print(f"Successfully loaded {len(loader.load())} pages from {pdf_path}")
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")
            continue

    if not all_docs:
        print("No documents were loaded successfully. Cannot create vector store.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Split {len(all_docs)} pages into {len(chunks)} text chunks.")

    # --- Embedding Model Selection ---
    if USE_OPENAI:
        embeddings = OpenAIEmbeddings() # Uses 'text-embedding-ada-002' by default
        print("Using OpenAIEmbeddings.")
    else:
        # Using a popular sentence-transformer model for embeddings from Hugging Face
        # This model runs locally or can be accessed via HuggingFace Hub if configured
        # For HuggingFace Hub, ensure HUGGINGFACEHUB_API_TOKEN is set.
        # You might need to install 'sentence-transformers' if running locally without the API:
        # pip install sentence-transformers
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Using HuggingFaceEmbeddings (all-MiniLM-L6-v2).")


    vector_store = FAISS.from_documents(chunks, embeddings)
    print("FAISS vector store created successfully.")
    return vector_store

def get_qa_chain(vector_store):
    """
    Creates a RetrievalQA chain, which is a core component of the RAG system.
    This chain combines a Language Model (LLM) with a retriever to answer questions
    based on the content of the vector store.

    Args:
        vector_store (FAISS): The FAISS vector store containing the document embeddings.

    Returns:
        RetrievalQA: An initialized LangChain RetrievalQA chain, or None if the
                     vector store is not provided.
    """
    if vector_store is None:
        print("Vector store is None. Cannot create QA chain.")
        return None

    # --- LLM Model Selection ---
    if USE_OPENAI:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        print("Using OpenAI ChatOpenAI (gpt-3.5-turbo).")
    else:
        # Using a Hugging Face Hub model for the LLM.
        # Ensure HUGGINGFACEHUB_API_TOKEN is set in your .env file.
        # You can choose other models from Hugging Face Hub, e.g.,
        # "google/flan-t5-large", "HuggingFaceH4/zephyr-7b-beta", "mistralai/Mistral-7B-Instruct-v0.2"
        # Be aware of model sizes and inference speeds on the free tier.
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large", # Changed back to Flan-T5 Large for better compatibility with HuggingFaceHub wrapper
            model_kwargs={"temperature": 0.7, "max_length": 512} # Adjusted max_length for Flan-T5 Large
        )
        print("Using HuggingFaceHub (google/flan-t5-large).")


    retriever = vector_store.as_retriever()

    prompt_template = """
    Use the following pieces of context to answer the user's question concisely and accurately.
    If you don't know the answer based on the provided context, just state that you don't know.
    Do not try to make up an answer.
    ----------------
    {context}
    Question: {question}
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print("RetrievalQA chain created successfully.")
    return qa_chain

def ask_question(qa_chain, question):
    """
    Asks a question using the provided RetrievalQA chain and returns the generated
    answer along with the source documents that were used.

    Args:
        qa_chain (RetrievalQA): The initialized LangChain RetrievalQA chain.
        question (str): The user's question.

    Returns:
        tuple: A tuple containing:
               - str: The answer generated by the LLM.
               - list: A list of source document chunks (Page objects) used for the answer.
                       Returns an empty list if no sources are found or an error occurs.
    """
    if qa_chain is None:
        return "Error: Document processing not complete. Please upload and process documents first.", []

    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        return answer, source_docs
    except Exception as e:
        print(f"An error occurred during question answering: {e}")
        return f"An error occurred while getting the answer: {e}", []

# Example of how you might use this locally for testing
if __name__ == "__main__":
    # --- Local Test Configuration ---
    # IMPORTANT: Place your PDF file(s) in the same directory as this rag_core.py file.
    # For example, if you have 'my_document.pdf' next to rag_core.py, list it here.
    sample_pdf_paths = ["sample_document.pdf"] # Replace with your actual PDF file names

    # Filter to only include PDFs that actually exist in the current directory
    existing_pdfs = [p for p in sample_pdf_paths if os.path.exists(p)]

    if not existing_pdfs:
        print("No existing PDF files found for testing in the current directory.")
        print("Please place your PDF file(s) (e.g., 'sample_document.pdf') in the same folder as rag_core.py.")
    else:
        try:
            print(f"Found PDF(s): {existing_pdfs}. Creating vector store...")
            vector_store = create_vector_store_from_pdfs(existing_pdfs)
            if vector_store:
                print("Creating QA chain...")
                qa_chain = get_qa_chain(vector_store)
                if qa_chain:
                    print("\n--- RAG System Ready for Testing ---")
                    print("You can now ask questions about your PDF content.")
                    
                    while True:
                        user_question = input("\nEnter your question (or type 'exit' to quit): ")
                        if user_question.lower() == 'exit':
                            print("Exiting RAG test.")
                            break
                        
                        print("\nAsking LLM...")
                        answer, sources = ask_question(qa_chain, user_question)
                        
                        print("\n--- Answer ---")
                        print(answer)
                        
                        if sources:
                            print("\n--- Sources Used ---")
                            for i, doc in enumerate(sources):
                                print(f"  Source {i+1} (Page: {doc.metadata.get('page', 'N/A')}, File: {os.path.basename(doc.metadata.get('source', 'N/A'))}):")
                                print(f"    Content Snippet: {doc.page_content[:300]}...") # Print first 300 chars of content
                        else:
                            print("\nNo specific source documents were retrieved for this answer.")
                else:
                    print("Failed to create QA chain. Check LLM configuration.")
            else:
                print("Failed to create vector store. Check PDF files or embedding model.")
        except ValueError as ve:
            print(f"Configuration Error: {ve}")
            print("Please ensure your HUGGINGFACEHUB_API_TOKEN or OPENAI_API_KEY is correctly set in your .env file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
