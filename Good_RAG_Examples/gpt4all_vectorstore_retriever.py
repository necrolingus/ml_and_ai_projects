from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

#setup variables
chroma_db_persist = 'c:/tmp/mytestChroma3_1/' #chroma will create the folders if they do not exist

#setup objects
gpt4all_embd = GPT4AllEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)


#read the text
loader = TextLoader("./mytext.txt")
docs = loader.load()
all_splits = text_splitter.split_documents(docs)


#create the vectorstore
vectorstore = Chroma.from_documents(documents=all_splits, persist_directory=chroma_db_persist, embedding=gpt4all_embd)

# Reuse an existing vectorstore use this and comment out the above
# vectorstore = Chroma(persist_directory=chroma_db_persist, embedding_function=gpt4all_embd) #this actually calls openAI, most likely ada
# vectorstore.get()


#ask a question
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("what countries do we do payments?")


#see the results
print(len(retrieved_docs))
print(retrieved_docs[0].page_content)
