class Template():
    def __init__(self, template_method):
        self.template_method = template_method

    def __call__(self, example, text):
        return self.template_method(example, text)


MISTRAL_TEMPLATE = Template(
    lambda example, text: """[INST] Based on the given Text please extract \
    information and format as json as given in the example. 
    Only use fields given in the example.
    Note: The example provided is solely for understanding the format. \
    Do NOT use the values from the example in your response.
    If you cant find the information for a field set the value to null.
    Dates should be formated YYYY, MM/YYYY or DD/MM/YYYY.
    Sort Arrays based on data descending.

    Example: ```{}````
    Text: ```{}```
    [/INST]""".format(example, text)
)