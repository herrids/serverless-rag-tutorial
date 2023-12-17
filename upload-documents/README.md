## TODO
# add csv example
# add faiss file to repository
# excpect prompt, instructions and hyperparameter in request
# integrate rag into handler
# create standalone Gradio app

# RunPod Worker for Mistral and RAG

## 📖 | Getting Started
Based on RunPod Documentation. Make sure you have Docker installed and an account for Docker Hub and RunPod.

1. Clone/Copy this repository.
2. Navigate to root and run `docker build --tag <username>/<repo_name>:<tag> .`. For example `docker build --tag edzhong/mistral:0.1 .`. This will take a while because the Model needs to be downloaded.
3. Deploy the created Image to your container registry with `docker push <username>/<repo_name>:<tag>`.
4. Go to RunPod console to define a template. Go to Serverless > Custom Template and click `New Template`. Give it a name, enter the image as `<username>/<repo_name>:<tag>` and choose your Docker credentials. Container Disk should be 15-20GB. Click Save Template
5. Create an Endpoint. Go to Serverless > Endpoint and press `New Enpoint`. Give it a name and select the template that was created. Select a GPU with at least 24GB VRAM. Set Active Workers to 0, max workers to the number of possible parallel requests (start with 1) and activate FlashBoot. Click Advanced and under Data Centers deselect all Non EU options. Then click `Deploy`.
6. After downloading the image the endpoint should be ready.

## Using the Endpoint.

The Endpoint has different POST addresses. They all need a Bearer Token that can be set in the settings. The RUN and RUNSYNC expect a JSON Body in this structure:

```json
{
    "input": {
        "prompt": "What is ther meaning of Life?"
    }
}
```

The RUNSYNC returns the response if it is created in 90 seconds. The RUN returns an ID with this ID the result can than get within 30 minutes under `/status/{job_id}``