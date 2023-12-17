from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os, shutil


def create_vector_store_index(file_path):

    file_path_split = file_path.split(".")
    file_type = file_path_split[-1].rstrip('/')

    if file_type == 'csv':
        print(file_path)
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()

    elif file_type == 'pdf':
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 128,)

        documents = text_splitter.split_documents(pages)
    
    file_output = "./db/faiss_index"

    try:
        vectordb = FAISS.load_local(file_output, OpenAIEmbeddings())
        vectordb.add_documents(documents)
    except:
        print("No vector store exists. Creating new one...")
        vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())

    vectordb.save_local(file_output)

    return "Vector store index is created."


def upload_and_create_vector_store(files):
    current_folder = os.getcwd()
    data_folder = os.path.join(current_folder, "data")

    # Create the directory if it doesn't exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    index_success_msg = "No new indices added."

    for file in files:
        # Save each file to a permanent location
        file_path = file
        split_file_name = file_path.split("/")
        file_name = split_file_name[-1]
        permanent_file_path = os.path.join(data_folder, file_name)

        if os.path.exists(permanent_file_path):
            print(f"File {file_name} already exists. Skipping.")
            continue

        shutil.copy(file, permanent_file_path)

        # Access the path of the saved file
        print(f"File saved to: {permanent_file_path}")

        # Create an index for each file and store the success messages
        index_success_msg = create_vector_store_index(permanent_file_path)

    return index_success_msg
