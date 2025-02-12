from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from chatglm_llm import ChatGLM

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

chatglm = ChatGLM()
chatglm.load_model()

def init_knowledge_vector_store():
    # embeddings = HuggingFaceEmbeddings(model_name="/home/jensenzhang/workspace/langchain-ChatGLM/models/text2vec-base-chinese", )
    # loader = UnstructuredFileLoader(filepath, mode="elements")
    # docs = loader.load()

    # vector_store = FAISS.from_documents(docs, embeddings)
    # return vector_store

    loader = DirectoryLoader('./data/mcd/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    persist_directory = 'db'
    ## Here is the nmew embeddings being used
    embedding = instructor_embeddings

    vectordb = Chroma.from_documents(documents=texts, 
                                    embedding=embedding,
                                    persist_directory=persist_directory)
    return vectordb

def get_knowledge_based_answer(query, vector_store, chat_history=[]):
    # system_template = """基于以下内容，简洁和专业的来回答用户的问题。
    # 如果无法从中得到答案，请说 "不知道" 或 "没有足够的相关信息"，不要试图编造答案。答案请使用中文。
    # ----------------
    # {context}
    # ----------------
    # """
    # messages = [
    #     SystemMessagePromptTemplate.from_template(system_template),
    #     HumanMessagePromptTemplate.from_template("{question}"),
    # ]
    # prompt = ChatPromptTemplate.from_messages(messages)

    condese_propmt_template = """基于以下内容，简洁和专业的来回答用户的问题。
    如果无法从中得到答案，请说 "不知道" 或 "没有足够的相关信息"，不要试图编造答案。答案请使用中文。

    任务: 给一段对话和一个后续问题，将后续问题改写成一个独立的问题。确保问题是完整的，没有模糊的指代。
    ----------------
    聊天记录：
    {chat_history}
    ----------------
    后续问题：{question}
    ----------------
    改写后的独立、完整的问题："""
    new_question_prompt = PromptTemplate.from_template(condese_propmt_template)
    chatglm.history = chat_history
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    knowledge_chain = ConversationalRetrievalChain.from_llm(
        llm=chatglm,
        retriever=retriever,
        condense_question_prompt=new_question_prompt,
    )

    knowledge_chain.return_source_documents = True
    # knowledge_chain.top_k_docs_for_context = 10

    result = knowledge_chain({"question": query, "chat_history": chat_history})
    return result, chatglm.history



vector_store = init_knowledge_vector_store()
history = []
while True:
    query = input("Input your question 请输入问题：")
    resp, history = get_knowledge_based_answer(query=query,
                                                vector_store=vector_store,
                                                chat_history=history)
    print(resp['answer'])
