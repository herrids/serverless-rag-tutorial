# Import necessary libraries and modules
from langchain.llms import VLLM  # Language models
from langchain.vectorstores import FAISS  # Vector storage for efficient similarity search
from langchain.prompts import PromptTemplate  # For creating structured prompts
from langchain.chains import ConversationalRetrievalChain  # Chain for combining retrieval and generation
from langchain.embeddings import HuggingFaceEmbeddings  # Embedding models
import runpod, os  # Runpod for serverless deployment, os for environment variables

# Setting up base paths using environment variables
BASE_PATH = os.environ.get('BASE_PATH', '/runpod-volume')
MODEL_PATH = os.environ.get('MODEL_PATH', '/models')
DB_PATH = os.environ.get('DB_PATH', '/db')
MODEL_NAME = os.environ.get('MODEL_NAME', 'mistralai')
EMBEDDING_MODEL_PATH = os.environ.get('EMBEDDING_MODEL_PATH', '/sentence_transformers')

# Function to get the vector database for embeddings
def get_vectordb(directory):
    embeddings_model = HuggingFaceEmbeddings(model_name=directory)
    vectordb = FAISS.load_local(f"{BASE_PATH}/{DB_PATH}", embeddings_model)
    
    return vectordb

# Handler function to process incoming jobs
def handler(job):
    """ Handler function that will be used to process jobs. """
    # Extracting job input parameters
    job_input = job['input']

    # Retrieving various parameters from the job input
    instruction = job_input.get('instruction')
    question = job_input.get('question')
    chat_history = job_input.get('history')
    k_context = job_input.get('k_context')
    temperature = job_input.get('temperature')
    max_new_tokens = job_input.get('max_new_tokens')
    repetition_penalty = job_input.get('repetition_penalty')
    top_k = job_input.get('top_k')
    top_p = job_input.get('top_p')

    print("Job Input:", job_input)

    # Setting up the language model with specified parameters
    llm = VLLM(
        model=f"{MODEL_PATH}/{MODEL_NAME}",
        vllm_kwargs={"max_model_len": 8192},
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p
    )

    # Loading the vector database for retrieval
    vectordb = get_vectordb(f"{BASE_PATH}/{EMBEDDING_MODEL_PATH}")

    # Template for the structured prompt
    template = instruction + """<s>[INST]
        context:\n
        {context}\n
        data: {question}\n
        [/INST]"""

    # Setting up the prompt template
    QCA_PROMPT = PromptTemplate(
        input_variables=["instruction", "context", "question"], 
        template=template)

    # Creating a conversational retrieval chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs={"k": k_context}),
        combine_docs_chain_kwargs={"prompt": QCA_PROMPT},
        get_chat_history=lambda h: h,
        verbose=True)

    # Executing the retrieval-augmented generation
    res = qa(
        {
            'question': question,
            'chat_history': chat_history
        }
    )

    # Preparing the response
    ret = {
        "result": res['answer']
    }
    return ret

# Starting the serverless function with the handler
runpod.serverless.start({"handler": handler})
