import gc, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
import requests

VECTOR_STORE_URL = os.environ.get('VECTOR_STORE_URL')
MODEL_URL = os.environ.get('MODEL_URL')
BEARER_TOKEN = os.environ.get('BEARER_TOKEN')

def upload_documents_to_vector_store(file):

    documents = []

    file_type = file.split(".")[-1].rstrip('/')

    if file_type == 'pdf':
        doc = fitz.open(file)
        full_text = ""

        for page in doc:
            full_text += page.get_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap = 128,
        )

        documents = text_splitter.split_text(full_text)

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {BEARER_TOKEN}'
    }

    response = requests.post(VECTOR_STORE_URL, headers=headers, json={
        "input": {        
            "documents": documents,
            "file_output": "runpod-volume/db"
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
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {BEARER_TOKEN}'
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
    }

    # Make the POST request to the RunPod endpoint
    response = requests.post(MODEL_URL, json=data, headers=headers)

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