import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

os.getenv("GROQ_API_KEY")

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0,groq_api_key=os.getenv("GROQ_API_KEY"),model="llama-3.3-70b-versatile")
    def extract_jobs(self,cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ###SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTIONS:
            The scrapped texxt is from the careers page of a website.
            your job is to extract  thr job postings and return them in JSON format containing the 
            following keys; 'role','experience','skills' and 'discription'.
            only retur the valid JSON.
            ###VALID JSON 
            NO PREAMBLE  STICTLY
            NO extra text other than josn file:
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={'page_data': cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Content too big.Unable to parse jobs")
        return res if isinstance(res,list) else [res]
    def write_mail(self,job,links):
        prompt_email = PromptTemplate.from_template(
            """
            ###JOB DESCRIPTION:
            {job_description}
            ###INSTRUCTION:
            You are [YOUR NAME] , a business developement executive at Crom.ai.Crom.ai is an AI $ Software consultant company 
            the seamless integrtion of business process through automated tools.
            Over out experience,we have empowred numerous enterprises with tailored solutions.
            process optimization,cost reduction,and heightened overall efficiency.
            Your job is to write a cold email to the client regarding the job mentioned above describing the 
            capability in fulfilling their needs.
            Also add the most relevant ones from the following links showcase croms's portfoli :{link_list}
            Remember you are john ,BDE at Crom.ai
            Do not provide preamble.
            ###EMAIL (NO PREAMBLE):
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

