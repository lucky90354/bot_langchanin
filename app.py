from langchain.document_loaders import TextLoader
import textwrap
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

# loading the API key
import os
os.environ['HUGGING_FACE_HUB_API_KEY'] = "Import Your HUGGING_FACE_HUB_API_KEY"

loader = TextLoader("data.txt")
document = loader.load()

# number of pages
len(document)

document[0]


def wrap_text_preserve_newlines(text, width=110):
    
    
    lines = text.split('\n')
    
    
    
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    
    
    wrapped_text = '\n'.join(wrapped_lines)
    
    return wrapped_text

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)

len(docs)
embedding = HuggingFaceEmbeddings()
doc_search = FAISS.from_documents(docs, embedding)

query = "Who is vivek singh"
similar_docs = doc_search.similarity_search(query)

repo_id = "tiiuae/falcon-7b"
llm = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGING_FACE_HUB_API_KEY'],
                     repo_id=repo_id, model_kwargs={'temperature': 0.2, 'max_length': 1000})

chain = load_qa_chain(
    llm,
    chain_type="stuff",
)

query = "Tell me something about vivek's profile"
docResult = doc_search.similarity_search(query)
print(chain.run(input_documents=docResult, question=query))
