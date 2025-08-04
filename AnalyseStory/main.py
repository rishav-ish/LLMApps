from langchain_core.prompts import PromptTemplate
#from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pprint import pprint
from operator import itemgetter

import dotenv

dotenv.load_dotenv('../.env')

#model = GoogleGenerativeAI(model="gemini-1.5-pro")
model = ChatGroq(model="llama-3.3-70b-versatile")
str_parser = StrOutputParser()

story_prompts = PromptTemplate.from_template("Write a short story about {topic}")
story_chain = story_prompts | model | str_parser

analyis_prompt = PromptTemplate.from_template("Analyse the following story's mood:\n{story}")
analysis_chain = analyis_prompt | model | str_parser


story_with_analysis = story_chain | analysis_chain

enhanced_chain = RunnablePassthrough.assign(story = story_chain).assign(analysis=analysis_chain)

manual_chain = (
        RunnablePassthrough() | 
        {
            "story": story_chain,
            "topic": itemgetter("topic")
        } |
        RunnablePassthrough().assign(
            analysis = analysis_chain
        )
    )

simple_dict_chain = story_chain | {"analysis": analysis_chain}

if __name__ == '__main__':
    subject = input("Enter the subject for story\n")
    #result = story_with_analysis.invoke(subject)
    #result = enhanced_chain.invoke({"topic": subject})
    result = manual_chain.invoke({"topic": subject})
    #result = simple_dict_chain.invoke({"topic": subject})
    print(result.keys())
    print("\nAnalysis")
    pprint(result)
    
