from langchain.llms import VLLM
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings  
import runpod, os

MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/models/')
MODEL_NAME = os.environ.get('MODEL_NAME', 'mistralai')
EMBEDDING_MODEL_PATH = os.environ.get(
    'EMBEDDING_MODEL_PATH', 
    '/runpod-volume/models/sentence_transformers')

llm = VLLM(
    model=f"{MODEL_BASE_PATH}/{MODEL_NAME}",
    vllm_kwargs={"max_model_len": 8192}
)

def get_vectordb(directory):
    embeddings_model = HuggingFaceEmbeddings(model_name=directory)
    vectordb = FAISS.load_local("./db/faiss_index", embeddings_model)
    
    return vectordb

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    instruction = job_input.get('instruction')
    question = job_input.get('question')
    chat_history = job_input.get('history')
    k_context = job_input.get('k_context')

    print("Job Input:", job_input)

    vectordb = get_vectordb(EMBEDDING_MODEL_PATH)

    template = instruction + """<s>[INST]
        context:\n
        {context}\n
        data: {question}\n
        [/INST]"""

    QCA_PROMPT = PromptTemplate(
        input_variables=["instruction", "context", "question"], 
        template=template)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs={"k": k_context}),
        combine_docs_chain_kwargs={"prompt": QCA_PROMPT},
        get_chat_history=lambda h: h,
        verbose=True)

    res = qa(
        {
            'question': question,
            'chat_history': chat_history
        }
    )

    ret = {
        "result": res['answer']
    }
    return ret

runpod.serverless.start({"handler": handler})
