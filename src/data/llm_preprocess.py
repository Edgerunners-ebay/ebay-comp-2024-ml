import dspy
import ollama
from src.models import ExtractCarInformation


def preprocess_desc_data(ollama_host, desc_data):
    prompt = f"""Extract the snippets that list vehicle compatibility information from the following data,
    
    "{desc_data}"
    """
    client = ollama.Client(host=ollama_host)
    return client.generate(model="llama3.2:latest", prompt=prompt)


def preprocess_desc_data_groq(text):
    with dspy.settings.context(
        lm=dspy.LM(model="groq/llama3-8b-8192", max_tokens=8000)
    ):
        preprocess_pipe = dspy.Predict(ExtractCarInformation)
        return preprocess_pipe.forward(description_data=text)


def preprocess_desc_data_groq_w_name(text, _model):
    with dspy.settings.context(lm=dspy.LM(model=_model)):
        preprocess_pipe = dspy.Predict(ExtractCarInformation)
        return preprocess_pipe.forward(description_data=text)
