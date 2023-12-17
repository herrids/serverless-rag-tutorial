import gc, os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests

def upload_documents_to_vector_store(file):

    url = os.environ.get('VECTOR_STORE_URL')
    bearer_token = os.environ.get('BEARER_TOKEN')

    documents = []

    file_type = file.split(".")[-1].rstrip('/')
    
    if file_type == 'csv':
        loader = CSVLoader(file_path=file)
        documents = loader.load()

    elif file_type == 'pdf':
        loader = PyPDFLoader(file)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap = 128,
        )

        documents = text_splitter.split_documents(pages)
        serialized_documents = [doc.to_json() for doc in documents]
        print(serialized_documents)

    headers = {
        'Authorization': f'Bearer {bearer_token}'
    }

    response = requests.post(url, headers=headers, json={
        "input": {        
            "documents": serialized_documents,
            "file_output": "db"
            }
        })

    # Check response status and handle accordingly
    if response.status_code == 200:
        print("Document sent successfully.")
    else:
        print(f"Failed to send document: {response}")

    return "Upload completed."


def get_chat_history(inputs):

    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAssistant:{ai}")
    return "\n".join(res)

def add_text(history, text):

    history = history + [[text, None]]
    return history, ""

def bot(history,
        instruction="Use the following pieces of context to answer the question at the end. Generate the answer based on the given context only if you find the answer in the context. If you do not find any information related to the question in the given context, just say that you don't know, don't try to make up an answer. Keep your answer expressive.",
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.1,
        top_k=10,
        top_p=0.95,
        k_context=5,
        num_return_sequences=1,
        ):
                             
    chat_history_formatted = get_chat_history(history[:-1])

    # Define the RunPod endpoint URL and any necessary headers
    url = os.environ.get('MODEL_URL')
    headers = {
        'Content-Type': 'application/json',
        # Add any necessary headers like authentication tokens here
    }

    # Prepare the data for the POST request
    data = {
        'question': history[-1][0],
        'chat_history': chat_history_formatted,
        'instruction': instruction,
        'temperature': temperature,
        'max_new_tokens': max_new_tokens,
        'repetition_penalty': repetition_penalty,
        'top_k': top_k,
        'top_p': top_p,
        'k_context': k_context,
        'num_return_sequences': num_return_sequences
        # Add any other parameters required by the RunPod
    }

    # Make the POST request to the RunPod endpoint
    response = requests.post(url, json=data, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        res = response.json()
        history[-1][1] = res['answer']
    else:
        history[-1][1] = "Error: Unable to get response from RunPod"

    return history



def reset_sys_instruction(instruction):

    default_inst = "Use the following pieces of context to answer the question at the end. Generate the answer based on the given context only if you find the answer in the context. If you do not find any information related to the question in the given context, just say that you don't know, don't try to make up an answer. Keep your answer expressive."
    return default_inst

def clear_cuda_cache():
    gc.collect()
    return None