import runpod, os
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.vectorstores import FAISS
from langchain.schema.document import Document

MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/runpod-volume')
MODEL_NAME = os.environ.get('MODEL_NAME', 'all-MiniLM-L6-v2')

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    file_output = job_input.get('file_output')
    documents = job_input.get('documents')

    print("Job Input:", job_input)

    print(os.getcwd())

    for file in os.listdir("/runpod-volume"):
        print(f"{file}")

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=f"{MODEL_BASE_PATH}/{MODEL_NAME}",
        model_kwargs={'device':'cuda'})

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
