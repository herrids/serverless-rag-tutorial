from langchain.llms import VLLM
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings  
import runpod, os

MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/models/')
MODEL_NAME = os.environ.get('MODEL_NAME', 'mistralai')
EMBEDDING_MODEL_PATH = os.environ.get('EMBEDDING_MODEL_PATH', '/runpod-volume/models/sentence_transformers')

llm = VLLM(
    model=f"{MODEL_BASE_PATH}/{MODEL_NAME}"
)

def get_chat_history(inputs):

    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAssistant:{ai}")
    return "\n".join(res)

def create_vectordb(directory):
    embeddings_model = HuggingFaceEmbeddings(model_name=directory)
    vectordb = FAISS.load_local("./db/faiss_index", embeddings_model)
    
    return vectordb

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    instruction = job_input.get('instruction')
    history = job_input.get('history')
    sampling_params = job_input.get('sampling_params')

    print("Job Input:", job_input)

    vectordb = create_vectordb(EMBEDDING_MODEL_PATH)

    template = instruction + """
        context:\n
        {context}\n
        data: {question}\n
        """

    QCA_PROMPT = PromptTemplate(input_variables=["instruction", "context", "question"], template=template)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs={"k": sampling_params.get("k_context")}),
        combine_docs_chain_kwargs={"prompt": QCA_PROMPT},
        get_chat_history=lambda h: h,
        verbose=True
    )

    chat_history_formatted = get_chat_history(history[:-1])

    res = qa(
        {
            'question': history[-1][0],
            'chat_history': chat_history_formatted
        }
    )

    history[-1][1] = res['answer']

    ret = {
        "result": history
    }
    return ret

runpod.serverless.start({"handler": handler})
