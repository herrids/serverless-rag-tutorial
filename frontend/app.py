import gradio as gr
from utils import *


with gr.Blocks(gr.themes.Soft(primary_hue=gr.themes.colors.slate, secondary_hue=gr.themes.colors.purple)) as demo:
    with gr.Row():

        with gr.Column(scale=1, variant = 'panel'):
            gr.Markdown("## Upload Document & Select the Embedding Model")
            file = gr.File(type="filepath")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, variant='compact'):
                    vector_index_btn = gr.Button('Create vector store', variant='primary',scale=1)
                    vector_index_msg_out = gr.Textbox(show_label=False, lines=1,scale=1, placeholder="Creating vectore store ...")

            instruction = gr.Textbox(label="System instruction", lines=3, value="Use the following pieces of context to answer the question at the end by. Generate the answer based on the given context only.If you do not find any information related to the question in the given context, just say that you don't know, don't try to make up an answer. Keep your answer expressive.")
            reset_inst_btn = gr.Button('Reset',variant='primary', size = 'sm')

            with gr.Accordion(label="Text generation tuning parameters"):
                temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.1, step=0.05)
                max_new_tokens = gr.Slider(label="max_new_tokens", minimum=1, maximum=4096, value=1024, step=1)
                repetition_penalty = gr.Slider(label="repetition_penalty", minimum=0, maximum=2, value=1.1, step=0.1)
                top_k= gr.Slider(label="top_k", minimum=1, maximum=1000, value=10, step=1)
                top_p=gr.Slider(label="top_p", minimum=0, maximum=1, value=0.95, step=0.05)
                k_context=gr.Slider(label="k_context", minimum=1, maximum=15, value=5, step=1)

            vector_index_btn.click(upload_documents_to_vector_store,[file],vector_index_msg_out)
            reset_inst_btn.click(reset_sys_instruction,instruction,instruction)

        with gr.Column(scale=1, variant = 'panel'):

            chatbot = gr.Chatbot([], elem_id="chatbot",
                                label='Chatbox', height=725, )

            txt = gr.Textbox(label= "Question",lines=2,placeholder="Enter your question and press shift+enter ")

            with gr.Row():

                with gr.Column(scale=1):
                    submit_btn = gr.Button('Submit',variant='primary', size = 'sm')

                with gr.Column(scale=1):
                    clear_btn = gr.Button('Clear',variant='stop',size = 'sm')

            txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot,instruction,temperature,max_new_tokens,repetition_penalty,top_k,top_p,k_context], chatbot)
            submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
                bot, [chatbot,instruction,temperature, max_new_tokens,repetition_penalty,top_k,top_p,k_context], chatbot).then(
                    clear_cuda_cache, None, None
                )

            clear_btn.click(lambda: None, None, chatbot, queue=False)


if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
