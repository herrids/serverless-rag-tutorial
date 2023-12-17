import runpod, os
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/runpod-volume/models/sentence_transformers')
MODEL_NAME = os.environ.get('MODEL_NAME', 'all-MiniLM-L6-v2')

def get_model(model_name, directory):
    # Check if the model exists in the specified directory
    if not os.path.exists(directory):
        print(f"Model not found in {directory}. Downloading from Hugging Face.")

        model = SentenceTransformer(model_name)
        model.save(directory)
        return HuggingFaceEmbeddings(model_name=directory)
    else:
        print(f"Model found in {directory}. Loading model.")
        return HuggingFaceEmbeddings(model_name=directory)

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    file_output = job_input.get('file_output')
    documents = job_input.get('documents')

    print("Job Input:", job_input)

    # Initialize the embedding model
    embedding_model = get_model(MODEL_NAME, MODEL_BASE_PATH)

    # Try loading the existing vector store, or create a new one
    try:
        vectordb = FAISS.load_local(file_output, embedding_model)
        vectordb.add_texts(documents)
    except:
        print("No vector store exists. Creating new one...")
        vectordb = FAISS.from_texts(documents, embedding_model)

    vectordb.save_local(file_output)

    ret = {
        "result": "Vector store index is created."
    }
    return ret

runpod.serverless.start({"handler": handler})
