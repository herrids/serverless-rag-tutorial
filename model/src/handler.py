from templates import MISTRAL_TEMPLATE
from example import EXAMPLE, EXAMPLE_STRING
from vllm import LLM, SamplingParams
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import runpod, json, os

MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', '/runpod-volume/')
MODEL_NAME = os.environ.get('MODEL_NAME', 'mistralai')

llm = LLM(
    f"{MODEL_BASE_PATH}/{MODEL_NAME}",
    dtype="bfloat16"
    )

def remove_json_formatting(input_string):
    if input_string.startswith("```json") and input_string.endswith("```"):
        return input_string[len("```json"): -len("```")].strip()
    else:
        return input_string

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    prompt = MISTRAL_TEMPLATE(EXAMPLE_STRING, job_input['document'])

    print("Job Input:", job_input)

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.1
    )

    response = llm.generate(prompt, sampling_params)

    print("response:", response)

    response_text = response[0].outputs[0].text

    try:
        formatted_response = remove_json_formatting(response_text.strip())

        response_schemas = [ResponseSchema(name=key, description="") for key in list(EXAMPLE.keys())]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        output = output_parser.parse(formatted_response)
    except Exception as e1:
        print(e1)
        try:
            output = json.loads(formatted_response)
        except Exception as e2:
            output = response_text

    ret = {
        "result": output
    }
    return ret

runpod.serverless.start({"handler": handler})
