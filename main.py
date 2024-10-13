#Finalized
import os
import boto3
import argparse
import streamlit as st
from langchain_aws import ChatBedrock
from langchain.load import dumps, loads
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def download_dir(client, resource, dist, local='/tmp', bucket='your_bucket'):
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, subdir.get('Prefix'), local, bucket)
        for file in result.get('Contents', []):
            dest_pathname = os.path.join(local, file.get('Key'))
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            if not file.get('Key').endswith('/'):
                resource.meta.client.download_file(bucket, file.get('Key'), dest_pathname)
def format_Tdocs(docs, k = 10):
    return "\n\n".join(doc[0].page_content for doc in docs[0:k+1])
def createQuestionsList(text):
    return text.split("\n")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def reciprocal_rank_fusion(results, k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

def Setup(embedmodel):
    print('hello, setup starts')
    
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    bedrock=boto3.client(service_name="bedrock-runtime", region_name='ap-south-1')
    llm = ChatBedrock(model_id="meta.llama3-8b-instruct-v1:0",client=bedrock, model_kwargs={"temperature": 0.1, "top_p": 0.9})
    
    model_name = embedmodel
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs, show_progress=True, cache_folder="./Embeddings"
    )
    print("Embeddings downloaded")
    
    loader = DirectoryLoader(
        "./RawDoc",
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,is_separator_regex=False, separators="\n")
    splits = text_splitter.split_documents(documents)
    loader = DirectoryLoader(
        "./SupportingDoc",
        glob="*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,is_separator_regex=False, separators="\n")
    SDsplits = text_splitter.split_documents(documents)
    print('Files setup')
    
    Mdb = FAISS.load_local(
        folder_path="./FAISS_data",
        embeddings=hf_embeddings,
        allow_dangerous_deserialization = True
    )
    retriever = Mdb.as_retriever()
    
    Sdb = FAISS.load_local(
        folder_path="./FAISS_SDdata",
        embeddings=hf_embeddings,
        allow_dangerous_deserialization = True
    )
    retriever_SD = Sdb.as_retriever()
    
    bm25_retriever = BM25Retriever.from_documents(splits)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
    )
    print("Retriever setup")
    print("Setup finished")
    return bedrock, llm, splits, SDsplits, Sdb, Mdb, retriever, retriever_SD, ensemble_retriever

def Get_Reponse(user_query, chat_history, bedrock, llm, retriever_SD, ensemble_retriever):
    template = """You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines. Original question: {question}

    Since there are some infomation that should be added to the question in order to provide the best results to retrieve the documents from the vector database. 
    You are supposed to include technical keywords to further help the document retrieval process.
    As the LLM which will analysis these questions and the retrieved documents have no prior knowledge on the documents nor the question, make the question technical

    NOTE: Be Direct with your questions. 

    The required documents are below: 

    Required Documents: {context}"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)

    getMoreQuestions = ({"context": retriever_SD | format_docs, "question": RunnablePassthrough()} 
                        | prompt_rag_fusion 
                        | ChatBedrock(model_id="mistral.mixtral-8x7b-instruct-v0:1",client=bedrock, model_kwargs={"temperature": 0.1, "top_p": 0.9})
                        | StrOutputParser()  
                        | createQuestionsList)
    
    retrieval_chain_rag_fusion = getMoreQuestions | ensemble_retriever.map() | reciprocal_rank_fusion
    
    template = """Imagine you are a developer and give me answer to the question based on the context alone.
    Since infomation within the context might be not be common knowledge and be known to you, I have included another supporting document for you to understand the context even better:

    NOTE: Do not hallucinate. Answer based only on the logs provided. If something isn't explicitly mentioned within the logs, do not assume anything unless you have a strong reason to. 
    If incase of you are assuming, mention that fact along with your reason to assume something within the answer. In the case of assuming, draw conclusions based only from the supporting document and when doing so, explain your train of thought to conclude as such.

    question : {question}

    supporting document for you to make better sense of the question : {supporting_document}

    Logs: {context}
    """
    prompt = ChatPromptTemplate.from_template(template)

    final_rag_chain = (
        {"context": retrieval_chain_rag_fusion | format_Tdocs, 
         "supporting_document": retriever_SD | format_docs,
         "question": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser()
    )
    return final_rag_chain.invoke(user_query)

if __name__ == '__main__':
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if 'state_setup' not in st.session_state:
        parser = argparse.ArgumentParser()
        parser.add_argument("--embedmodel", type=str, default="BAAI/bge-base-en-v1.5", help="SD for SupportingDoc; RD for RawDocs")
        args = parser.parse_args()
        bedrock, llm, splits, SDsplits, Sdb, Mdb, retriever, retriever_SD, ensemble_retriever = Setup(embedmodel=args.embedmodel)
        st.session_state.state_setup = [bedrock, llm, splits, SDsplits, Sdb, Mdb, retriever, retriever_SD, ensemble_retriever]
    if 'stop' not in st.session_state:
        st.session_state['stop'] = False

    st.set_page_config(page_title="RAGSync: Chat")
    st.title('RAGSync Bot')
    
    if st.button('Stop the bot?'):
        st.session_state['stop'] = True
    if st.session_state['stop']:
        st.write('Stopping the app...')
        st.stop()    

    for message in st.session_state.chat_history: 
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message('YBot'):
                st.markdown(message.content)

    user_query = st.chat_input("Your query")
    if user_query is not None and user_query != '':
        st.session_state.chat_history.append(HumanMessage(user_query))
    
        with st.chat_message('Human'):
            st.markdown(user_query)
        
        with st.chat_message("YBot"):
            ai_message = Get_Reponse(user_query, st.session_state.chat_history, st.session_state.state_setup[0],  st.session_state.state_setup[1], st.session_state.state_setup[7], st.session_state.state_setup[8])
            st.markdown(ai_message)
            st.session_state.chat_history.append(AIMessage(ai_message))
