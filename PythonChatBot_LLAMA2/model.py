from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as ch

DB_P = 'vectorstore/db_faiss'

custom_pt = """Use the following information to answer user's questions.
if you are unaware of the answer, state that you do not know, do not try to make up your own answer.

Context: {context}
Question: {question}

Only return helpful answer and nothing more.
Helpful answer:
"""

def set_custom_prompt():
    
    prompt = PromptTemplate(template=custom_pt, input_variables=['context','question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':prompt})
    return qa_chain

def load_llm():
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_P, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

@ch.on_chat_start
async def start():
    chain = qa_bot()
    msg = ch.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to your PDF Bot. What is your query?"
    await msg.update()

    ch.user_session.set("chain", chain)

@ch.on_message
async def main(message):
    chain = ch.user_session.get("chain") 
    cb = ch.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await ch.Message(content=answer).send()