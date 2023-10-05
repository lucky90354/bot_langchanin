from langchain.document_loaders import TextLoader
import textwrap
from langchain.text_splitter import CharacterTextSplitter
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
loader =TextLoader("data.txt")
document =loader.load()

#preprocessing



def wrap_text_preserve_newlines(text, width=110):
    #split the input text into the lines based  on the newline charcters
    
    lines = text.split('\n')

    #wrap each line indivisually

    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    #join the wrapped lines back together using new line charcters

    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

#Text splitting

text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
docs = text_splitter.split_documents(document)

print(docs[0])
print(len(docs))

#Embedding
 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
embedding = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embedding)

#Q-A

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8, "max_length":512})

chain = load_qa_chain(llm, chain_type="stuff")

query = "what is langchain?"

docResult = db.similarity_search(query)
print(chain.run(input_ducument = docResult, question = query))