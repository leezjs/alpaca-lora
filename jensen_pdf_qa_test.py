import os

os.environ['OPENAI_API_KEY'] = "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"
os.environ['OPENAI_API_BASE'] = "https://open2.aiproxy.xyz/v1"

from PyPDF2 import PdfReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.document_loaders import TextLoader

def read_pdf_files(directory):
    raw_text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            raw_text += read_pdf(file_path)
    return raw_text

def read_pdf(file_path):
    with open(file_path, "rb") as file:
        raw_text = ""
        pdf_reader = PdfReader(file)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                # print(text)
                raw_text += text
        return raw_text
    
# raw_text = read_pdf_files("./data/mcd/")
# text_splitter = CharacterTextSplitter(        
#     separator = "\n",
#     chunk_size = 1000,
#     chunk_overlap  = 200, #striding over the text
#     length_function = len,
# )
# texts = text_splitter.split_text(raw_text)

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("./data/mcd/MCoffee.pdf")
pages = loader.load_and_split()

embeddings = HuggingFaceEmbeddings()
docsearch = FAISS.from_documents(pages, embeddings)
# docs = docsearch.similarity_search(query)

from langchain.llms import OpenAI

# chain = load_qa_chain(OpenAI(), chain_type="map_rerank", return_intermediate_steps=True)
# query = "who is openai?"
# docs = docsearch.similarity_search(query,k=10)
# results = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
# print( results )

from langchain.chains import RetrievalQA

# set up FAISS as a generic retriever 
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})

# create the chain to answer questions 
rqa = RetrievalQA.from_chain_type(llm=OpenAI(), 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)

# query = "What does gpt-4 mean for creativity?"
# query = "奶铁该怎么制作?"
# query = "麦咖啡的特色是什么?"
query = "麦咖啡的制作流程是什么样的？"
print( rqa(query)['result'] )
