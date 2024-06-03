

# import openai
# import langchain
# import pinecone 
# from langchain.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import  Chroma
# from langchain.llms import OpenAI
# from dotenv import load_dotenv
# load_dotenv()
# import os

import requests
import json
BASE= "http://127.0.0.1:5000/"
query="what is required in this assignment?"
response = requests.post(BASE+"Chat/hello")
print(response)





# #read document 

# def read_pdf(directory_path):
#     file_loader= PyPDFDirectoryLoader(directory_path)
#     documents=file_loader.load()
#     return documents
# doc = read_pdf('docs/')
# print(len(doc))

# # devide the docs into ckunks
# def chunk_data(docs, chunk_size=1000, chunk_overlap=50):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
#     doc= splitter.split_documents(docs)
#     return doc

# documents=chunk_data(docs=doc)


# embeddings=OpenAIEmbeddings(api_key=os.environ['OPEN_API_KEY'])

# vectordb=Chroma.from_documents(documents=documents,embedding=embeddings, persist_directory='db')
# vectordb.persist()
# vectordb=None
# vectordb=Chroma(persist_directory="db",embedding_function=embeddings)

# ## Make retirever

# retriever = vectordb.as_retriever()
# docs=retriever.get_relevant_documents("what is required in this assignment?")
# retriever= vectordb.as_retriever(search_kwargs={'k':2})

# #make a chaim
# from langchain.chains import RetrievalQA

# from langchain_openai import ChatOpenAI

# llm=ChatOpenAI(api_key=os.environ['OPEN_API_KEY'],model_name='gpt-4')
# llm2=ChatOpenAI(api_key=os.environ['OPEN_API_KEY'],model_name='gpt-3.5-turbo')
# # Define the prompt template
# from langchain import PromptTemplate
# prompt_template = """
# You are an AI that answers questions based on the given search results. Only use the information provided in the search results to construct your response. Do not use any external knowledge.
# Search Results:
# {documents}

# Question:
# {query}
# Answer:
# """
# query="what is required in this assignment?"
# prompt=PromptTemplate.from_template(prompt_template)
# prompt.format(documents=docs, query=query)


# from langchain import LLMChain
# llm_chain=LLMChain(prompt=prompt,llm=llm)
# print(llm_chain.run({'query':query,'documents':docs}))













#create the chaiinn to answer questions
# qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                         chain_type='stuff',
#                                         retriever=retriever,
#                                         return_source_documents=True)
# llm_response= qa_chain(query)
# print(llm_response['result'])

# qa_chain = RetrievalQA.from_chain_type(llm=llm2,
#                                         chain_type='stuff',
#                                         retriever=retriever,
#                                         return_source_documents=True)
# llm_turbo_response=qa_chain(query)
# print(llm_turbo_response['result'])


# import replicate
# from replicate.client import Client

# replicate = Client(api_token="r8_Be7LYLyakwl3pQdKUA9E6NuWXOlCx9u46SPb5")


# output = replicate.run(
#     "joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",
    
#     input={"prompt": query}
# )
# print("".join(output))

# from langchain_community.llms import HuggingFaceHub


# api_token = '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'
# falcon_llm = HuggingFaceHub(huggingfacehub_api_token=api_token, repo_id='tiiuae/falcon-40b-instruct')


# from llama_index.llms.together import TogetherLLM
# llm= TogetherLLM(
#     model= 'togethercomputer/llama-2-70b-chat',
#     temperature=0.1,
#     max_tokens=1024
# )

# qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
# llm_response= qa_chain(query)
# print(llm_response['result'])