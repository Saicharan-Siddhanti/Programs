from langchain_ollama import OllamaLLM

model = OllamaLLM(model="llama2")

result = model.invoke("What is the meaning of transformers in deep learning?")
print(result)