# Import necessary libraries and modules
import runpod, os
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Define environment variables for model paths and names
MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/runpod-volume/models/sentence_transformers')
MODEL_NAME = os.environ.get('MODEL_NAME', 'all-MiniLM-L6-v2')

def get_model(model_name, directory):
    """
    Function to get the sentence transformer model.
    Checks if the model exists in the specified directory. 
    If not, downloads it from Hugging Face and saves it locally.
    """
    if not os.path.exists(directory):
        print(f"Model not found in {directory}. Downloading from Hugging Face.")

        # Download and save the model from Hugging Face
        model = SentenceTransformer(model_name)
        model.save(directory)
        return HuggingFaceEmbeddings(model_name=directory)
    else:
        # Load the model from the local directory if it exists
        print(f"Model found in {directory}. Loading model.")
        return HuggingFaceEmbeddings(model_name=directory)

def handler(job):
    """
    Handler function to process jobs in runpod.
    It creates or updates a vector store index with document embeddings.
    """
    job_input = job['input']
    file_output = job_input.get('file_output')
    documents = job_input.get('documents')

    print("Job Input:", job_input)

    # Load or download the embedding model
    embedding_model = get_model(MODEL_NAME, MODEL_BASE_PATH)

    try:
        # Load the existing vector store (FAISS index) and add new document embeddings
        vectordb = FAISS.load_local(file_output, embedding_model)
        vectordb.add_texts(documents)
    except:
        # If no vector store exists, create a new one from the documents
        print("No vector store exists. Creating new one...")
        vectordb = FAISS.from_texts(documents, embedding_model)

    # Save the updated or new vector store locally
    vectordb.save_local(file_output)

    # Return a message indicating successful creation or update of the vector store
    ret = {
        "result": "Vector store index is created."
    }
    return ret

# Start the runpod serverless function with the defined handler
runpod.serverless.start({"handler": handler})
