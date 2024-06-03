from flask import Flask, request, jsonify, send_from_directory
from flask_restful import Api, Resource
import openai
import langchain
import pinecone 
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import  Chroma
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain import LLMChain
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from langchain import HuggingFacePipeline
from langchain.model_laboratory import ModelLaboratory

app = Flask(__name__)
api = Api(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
# cors = CORS(app, resources={r"/chat":  {
#         "origins": "*",             # Allows all origins
#         "methods": ["*"],           # Allows all methods
#         "allow_headers": ["*"],     # Allows all headers
#         "expose_headers": [],       # No exposed headers
#         "max_age": 0                # Max age
#     }})


class Chat4(Resource):
    def __init__(self):
        load_dotenv()
        self.embeddings = OpenAIEmbeddings(api_key=os.environ['OPEN_API_KEY'])

    def read_pdf(self, directory_path):
        file_loader = PyPDFDirectoryLoader(directory_path)
        documents = file_loader.load()
        return documents

    def chunk_data(self, docs, chunk_size=1000, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc_chunks = splitter.split_documents(docs)
        return doc_chunks

    def heuristic_evaluation(self, response, query):
        """Heuristic evaluation of the response."""
        score = 0
        criteria = {
              "coherence": lambda response: 1 if len(response) > 0 else 0,
              "relevance": lambda response, query: 1 if query in response else 0,
              "informativeness": lambda response: 1 if len(response.split()) > 10 and "..." not in response else 0,
              "accuracy": lambda response: 1 if "according to" in response else 0,  # Placeholder for more accurate check
              "conciseness": lambda response: 1 if len(response.split()) < 100 else 0,
        }

        score += criteria["coherence"](response)
        score += criteria["relevance"](response, query)
        score += criteria["informativeness"](response)
        score += criteria["accuracy"](response)
        score += criteria["conciseness"](response)

        return score / len(criteria)

    def post(self, query):
        # Read and chunk documents
        doc = self.read_pdf('docs/')
        documents = self.chunk_data(docs=doc)
        
        # Create vector store
        vectordb = Chroma.from_documents(documents=documents, embedding=self.embeddings, persist_directory='db')
        vectordb.persist()
        vectordb = Chroma(persist_directory="db", embedding_function=self.embeddings)

        # Retrieve relevant documents
        retriever = vectordb.as_retriever(search_kwargs={'k': 2})
        docs = retriever.get_relevant_documents(query)

        # Define prompt template
        prompt_template = """
        You are an AI that answers questions based on the given search results. Only use the information provided in the search results to construct your response. Do not use any external knowledge.
        Search Results:
        {documents}

        Question:
        {query}
        Answer:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        formatted_prompt = prompt.format(documents="\n\n".join([doc.page_content for doc in docs]), query=query)

        # Define language model
        llm = ChatOpenAI(api_key=os.environ['OPEN_API_KEY'], model_name='gpt-4')
      
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run({'query': query, 'documents': "\n\n".join([doc.page_content for doc in docs])})
        print(response)

        # Evaluate the response
        score = self.heuristic_evaluation(response, query)

        # Return the response and evaluation score
        return jsonify({"response": response, "score": score})
api.add_resource(Chat4,"/Chat4/<string:query>")
class Chat3(Resource):
    def __init__(self):
        load_dotenv()
        self.embeddings = OpenAIEmbeddings(api_key=os.environ['OPEN_API_KEY'])

    def read_pdf(self, directory_path):
        file_loader = PyPDFDirectoryLoader(directory_path)
        documents = file_loader.load()
        return documents

    def chunk_data(self, docs, chunk_size=1000, chunk_overlap=50):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc_chunks = splitter.split_documents(docs)
        return doc_chunks
    
    def heuristic_evaluation(self, response, query):
        """Heuristic evaluation of the response."""
        score = 0
        criteria = {
              "coherence": lambda response: 1 if len(response) > 0 else 0,
              "relevance": lambda response, query: 1 if query in response else 0,
              "informativeness": lambda response: 1 if len(response.split()) > 10 and "..." not in response else 0,
              "accuracy": lambda response: 1 if "according to" in response else 0,  # Placeholder for more accurate check
              "conciseness": lambda response: 1 if len(response.split()) < 100 else 0,
        }

        score += criteria["coherence"](response)
        score += criteria["relevance"](response, query)
        score += criteria["informativeness"](response)
        score += criteria["accuracy"](response)
        score += criteria["conciseness"](response)


        return score / len(criteria)
   
    def post(self, query):
        # Read and chunk documents
        doc = self.read_pdf('docs/')
        documents = self.chunk_data(docs=doc)
        
        # Create vector store
        vectordb = Chroma.from_documents(documents=documents, embedding=self.embeddings, persist_directory='db')
        vectordb.persist()
        vectordb = Chroma(persist_directory="db", embedding_function=self.embeddings)

        # Retrieve relevant documents
        retriever = vectordb.as_retriever(search_kwargs={'k': 2})
        docs = retriever.get_relevant_documents(query)

        # Define prompt template
        prompt_template = """
        You are an AI that answers questions based on the given search results. Only use the information provided in the search results to construct your response. Do not use any external knowledge.
        Search Results:
        {documents}

        Question:
        {query}
        Answer:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        formatted_prompt = prompt.format(documents="\n\n".join([doc.page_content for doc in docs]), query=query)

        # Define language model
        llm = ChatOpenAI(api_key=os.environ['OPEN_API_KEY'], model_name='gpt-3.5-turbo')
      
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run({'query': query, 'documents': "\n\n".join([doc.page_content for doc in docs])})
        print(response)

        # Evaluate the response
        score = self.heuristic_evaluation(response, query)

        # Return the response and evaluation score
        return jsonify({"response": response, "score": score})
api.add_resource(Chat3,"/Chat3/<string:query>")

if __name__ == '__main__':
    app.run(debug=True) 
