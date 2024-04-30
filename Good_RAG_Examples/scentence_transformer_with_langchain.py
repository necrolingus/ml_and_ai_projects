from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



#setup variables
chroma_db_persist = 'c:/tmp/mytestChroma3_1/' #chroma will create the folders if they do not exist


#setup objects
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80, add_start_index=True)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


#read the text
loader = TextLoader("./mytext.txt")
docs = loader.load()
all_splits = text_splitter.split_documents(docs)


#create the vectorstore
vectorstore = Chroma.from_documents(documents=all_splits, persist_directory=chroma_db_persist, embedding=embedding_function)


query = "what moneytary value does nutun process?"
retrieved_docs = vectorstore.similarity_search(query)

# print results
print(len(retrieved_docs))
print(retrieved_docs[0].page_content)





