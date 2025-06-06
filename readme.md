RAG Document Q&A System
This project implements a Retrieval-Augmented Generation (RAG) system that allows you to ask questions about the content of your PDF documents. It leverages Large Language Models (LLMs) to provide accurate answers grounded in your specific knowledge base.
✨ Features
PDF Document Ingestion: Load and process multiple PDF files.
Intelligent Text Chunking: Documents are split into optimized chunks for efficient retrieval.
Semantic Search: Uses embeddings to find the most semantically relevant document sections.
LLM-Powered Answering: Generates coherent and contextual answers using an LLM (either OpenAI or Hugging Face).
Source Attribution: Provides the specific document pages/files that were used to formulate the answer, enhancing transparency.
Local Test Version: Includes a command-line interface for easy local testing without a full web UI.
🚀 Technical Stack
Python: The core programming language.
LangChain: Framework for building LLM applications.
langchain-community: Provides various integrations like PyPDFLoader and FAISS.
langchain-openai: (Optional) For OpenAI LLM and embeddings.
huggingface_hub: (Default) For accessing Hugging Face models (LLM and embeddings).
pypdf: For parsing PDF documents.
tiktoken: Used for token counting (especially with OpenAI models).
faiss-cpu: An efficient library for similarity search, used as our local vector database.
python-dotenv: For securely loading environment variables (like API keys).
sentence-transformers: (Dependency for HuggingFaceEmbeddings) For generating high-quality text embeddings.
🛠️ Setup Instructions
Follow these steps to get the RAG system running on your local machine.
1. Clone the Repository (or create the project structure)
mkdir rag_document_qa
cd rag_document_qa


2. Create a Virtual Environment (Recommended)
python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


3. Install Dependencies
Create a file named requirements.txt in your project directory with the following content:
langchain-community
langchain-openai
pypdf
tiktoken
faiss-cpu
streamlit
python-dotenv
sentence-transformers
huggingface_hub


Then, install them:
pip install -r requirements.txt


4. Configure API Keys (.env file)
You need an API key for an LLM provider. This project supports both Hugging Face Hub and OpenAI.
Create a file named .env in the root of your rag_document_qa directory.
Option A: Using Hugging Face Hub (Recommended for free tier)
Go to huggingface.co/settings/tokens.
Sign up/Log in and generate a new access token (with a "read" role).
Add the following line to your .env file, replacing the placeholder with your actual token:
HUGGINGFACEHUB_API_TOKEN="hf_YOUR_HUGGING_FACE_TOKEN_HERE"


Option B: Using OpenAI (Requires a paid account)
Go to platform.openai.com/account/api-keys.
Create a new secret key.
Add the following line to your .env file, replacing the placeholder with your actual key:
OPENAI_API_KEY="your_openai_api_key_here"


Note: The rag_core.py script is configured to prioritize HUGGINGFACEHUB_API_TOKEN if both are present.
5. Place Your PDF Documents
Place the PDF file(s) you want to query directly into the rag_document_qa directory (the same folder where rag_core.py is located).
Important: In rag_core.py, locate the sample_pdf_paths list within the if __name__ == "__main__": block and update it with the exact filenames of your PDFs.
# Example:
sample_pdf_paths = ["my_report.pdf", "another_document.pdf"]


6. Add rag_core.py
Save the rag_core.py code (from the latest immersive artifact) into a file named rag_core.py in your project directory.
🏃 How to Run the Local Test Version
Once you have completed the setup:
Open your terminal or command prompt.
Navigate to your rag_document_qa directory.
Activate your virtual environment:
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


Run the script:
python rag_core.py


The script will then:
Detect and load your specified PDF files.
Create a vector store and a QA chain.
Prompt you to enter questions in the terminal.
Provide answers and the source document snippets.
Type exit to quit the application.
🐛 Troubleshooting
ValueError: Neither HUGGINGFACEHUB_API_TOKEN nor OPENAI_API_KEY found.
Solution: Ensure your .env file is correctly placed in the same directory as rag_core.py and contains at least one of the required API keys (e.g., HUGGINGFACEHUB_API_TOKEN="..."). Double-check for typos or missing quotes.
'InferenceClient' object has no attribute 'post'
Solution: This error often relates to the specific Hugging Face model or huggingface_hub library version.
Ensure huggingface_hub is installed: Run pip install huggingface_hub.
Verify repo_id: The current rag_core.py uses "google/flan-t5-large". While generally compatible, some models might have specific inference requirements. If the issue persists, try a different repo_id for the LLM from Hugging Face Hub that is known for text generation and works well with the Inference API (e.g., "HuggingFaceH4/zephyr-7b-beta" or "mistralai/Mistral-7B-Instruct-v0.2"). Remember to adjust max_length in model_kwargs if you change the model.
Check your Hugging Face Token: Ensure your HUGGINGFACEHUB_API_TOKEN is correct and has the necessary permissions (read access is usually sufficient).
💡 Future Enhancements
Streamlit Web UI: Integrate this core logic into a Streamlit app.py for a user-friendly web interface (as discussed in previous steps).
Persistent Vector Store: Save the FAISS index to disk so documents don't need to be re-processed every time the script runs.
Support for More Document Types: Extend the loader to handle .docx, .txt, etc.
Advanced Chunking Strategies: Implement more sophisticated text splitting techniques.
Evaluation Metrics: Add functionality to evaluate the quality of answers.
Chat History: Maintain a conversational history for multi-turn interactions.
