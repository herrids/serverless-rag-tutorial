import runpod, os
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.vectorstores import FAISS

MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/runpod-volume/')
MODEL_NAME = os.environ.get('MODEL_NAME', 'all-MiniLM-L6-v2')

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    file_output = job_input.get('file_output')
    documents = job_input.get('documents')

    print("Job Input:", job_input)

    print(f"{MODEL_BASE_PATH}{MODEL_NAME}")

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(f"{MODEL_BASE_PATH}{MODEL_NAME}")

    # Try loading the existing vector store, or create a new one
    try:
        vectordb = FAISS.load_local(file_output, embedding_model)
        vectordb.add_documents(documents)
    except:
        print("No vector store exists. Creating new one...")
        vectordb = FAISS.from_documents(documents, embedding_model)

    vectordb.save_local(file_output)

    ret = {
        "result": "Vector store index is created."
    }
    return ret

runpod.serverless.start({"handler": handler})
