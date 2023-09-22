import os
import streamlit as st
from apikey import apikey
from langchain.llms import OpenAI
from langchain.prompts  import PromptTemplate
from langchain.chains  import SequentialChain, LLMChain
from langchain.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = apikey

st.title("👾 GPT Content Creator")
prompt = st.text_input("Enter your prompt here")

title_template = PromptTemplate(input_variables=['topic'], template='write me a youtube video title regarding {topic}')

script_template = PromptTemplate(input_variables=['title', 'wikipedia_research'], template='write me a youtube script about {title} based on {wikipedia_research}')

title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm = llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm = llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

if prompt:
    title = title_chain.run(prompt)
    wiki_research =wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research = wiki_research)
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)
    
    with st.expander('Script History'):
        st.info(script_memory.buffer)
    
    with st.expander('Wikipedia Research History'):
        st.info(wiki_research)