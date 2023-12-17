import json

"""
If you want to use the Model to create and return a json or a csv it is good 
practise todo something called One-Shot Learning. This means we give it exactly 
one example on what output we expect. This example we than add to the prompt.
In this file I add a general json and csv example that can be altered to someones
desires.
"""

EXAMPLE_JSON = {
    "first": "Tom", 
    "lastname": "Miller",
    "Email Adresse": "t.miller@gmail.com",
    "Date of birth": "03rd August 1998",
}

EXAMPLE_JSON_STRING = json.dumps(EXAMPLE_JSON)

