from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate


llm = OllamaLLM(model="llama3.2")


result = llm.invoke("Tell 2 jokes about dragons")

embed = OllamaEmbeddings(model="llama3.2")

vector = embed.embed_query(result)

print(vector[:5])