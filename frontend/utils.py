import gc, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF library for working with PDFs
import requests
from instruction import DEFAULT_INSTRUCTION

# Fetching environment variables for URLs and tokens
VECTOR_STORE_URL = os.environ.get('VECTOR_STORE_URL')
MODEL_URL = os.environ.get('MODEL_URL')
BEARER_TOKEN = os.environ.get('BEARER_TOKEN')

def upload_documents_to_vector_store(file):
    # Function to upload documents to a vector store

    documents = []

    # Extracting file type from the file path
    file_type = file.split(".")[-1].rstrip('/')

    # Checking if the file is a PDF
    if file_type == 'pdf':
        doc = fitz.open(file)  # Opening the PDF file
        full_text = ""

        # Extracting text from each page of the PDF
        for page in doc:
            full_text += page.get_text()

        # Initializing a text splitter to break text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 512,
            chunk_overlap = 128,
        )

        # Splitting the full text into smaller documents
        documents = text_splitter.split_text(full_text)

    # Setting up headers for the HTTP request
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {BEARER_TOKEN}'
    }

    # Sending a POST request to the vector store URL with the documents
    response = requests.post(VECTOR_STORE_URL, headers=headers, json={
        "input": {        
            "documents": documents,
            "file_output": "runpod-volume/db"
            }
        })

    # Checking the response status
    if response.status_code == 200:
        print("Document sent successfully.")
    else:
        print(f"Failed to send document: {response}")

    return "Upload completed."

def get_chat_history(inputs):
    # Function to format the chat history

    res = []
    for human, ai in inputs:
        # Adding each exchange to the formatted history
        res.append(f"Human:{human}\nAssistant:{ai}")
    return "\n".join(res)

def add_text(history, text):
    # Function to add new text to the chat history

    history = history + [[text, None]]
    return history, ""

def bot(history,
        instruction=DEFAULT_INSTRUCTION,
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.1,
        top_k=10,
        top_p=0.95,
        k_context=5
        ):
    # Main function to interact with an AI model

    # Formatting the chat history for the AI model
    chat_history_formatted = get_chat_history(history[:-1])

    # Setting up headers for the HTTP request
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {BEARER_TOKEN}'
    }

    # Preparing the data payload for the request
    data = {
        "input": {
            'question': history[-1][0],
            'chat_history': chat_history_formatted,
            'instruction': instruction,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'repetition_penalty': repetition_penalty,
            'top_k': top_k,
            'top_p': top_p,
            'k_context': k_context,
        }
    }

    # Sending a POST request to the AI model URL
    response = requests.post(MODEL_URL, json=data, headers=headers)

    # Processing the response
    if response.status_code == 200:
        res = response.json()
        history[-1][1] = res['output']['result']
    else:
        history[-1][1] = "Error: Unable to get response from RunPod"

    return history

def reset_sys_instruction():
    # Function to reset system instruction to the default
    return DEFAULT_INSTRUCTION;

def clear_cuda_cache():
    # Function to clear the CUDA memory cache (for GPU memory management)
    gc.collect()
    return None
