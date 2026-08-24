from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
def generate_pet_name():
    llm=OpenAI(temperature=0)
    name=llm.invoke("i have a cat can you suggest me 10 beautiful names")
    return name
if __name__=="__main__":
    print(generate_pet_name())