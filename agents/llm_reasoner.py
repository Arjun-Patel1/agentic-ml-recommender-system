from langchain_community.llms import Ollama

llm = Ollama(model="mistral")

def reason(prompt: str) -> str:
    return llm.invoke(prompt)
