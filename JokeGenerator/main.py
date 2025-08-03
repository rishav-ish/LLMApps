from langchain_core import output_parsers
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv('../.env')


prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatGroq(model="llama-3.3-70b-versatile")

output_parser = StrOutputParser()

chain = prompt | model | output_parser


if __name__ == "__main__":
    topic = input("Please provide topic for the joke\n")
    result = chain.invoke({"topic": topic})
    print(result)
