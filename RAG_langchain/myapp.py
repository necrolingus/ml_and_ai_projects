#https://betterprogramming.pub/building-a-support-bot-for-your-website-using-python-openai-3cde0a6b5e91

#https://python.langchain.com/docs/use_cases/question_answering/quickstart
#pip install --upgrade langchain
#pip install --upgrade langchain-community
#pip install --upgrade langchainhub
#pip install --upgrade langchain-openai
#pip install --upgrade chromadb 
#pip install --upgrade bs4


import sys, os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from datetime import datetime

openai_key = os.environ["OPENAI_CHATGPT_API_KEY"]
chroma_db_persist = 'c:/tmp/mytestChroma11/' #chroma will create the folders if they do not exist
embedding_openai = OpenAIEmbeddings(openai_api_key = openai_key)
our_query = "What are the approaches to Task Decomposition?"

# TO LOAD WEBSITE CONTENT
#Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()


#Our loaded document is over 42k characters long. 
#This is too long to fit in the context window of many models. 
#Even for those models that could fit the full post in their context window, 
#models can struggle to find information in very long inputs.
print(type(docs[0]))
print(docs[0].page_content[:500])
#print(len(docs[0].page_content)) #to see the keys and page content


#To handle this we’ll split the Document into chunks for embedding and vector storage. 
#This should help us retrieve only the most relevant bits of the blog post at run time.
#In this case we’ll split our documents into chunks of 1000 characters with 200 characters of overlap between chunks.
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)
print(len(all_splits))
print(len(all_splits[0].page_content))
print(all_splits[10].metadata)



# Now we need to index our 66 text chunks so that we can search over them at runtime. 
# The most common way to do this is to embed the contents of each document split and insert these embeddings 
# into a vector database (or vector store). 
# When we want to search over our splits, we take a text search query, embed it, 
# and perform some sort of “similarity” search to identify the stored splits with the most 
# similar embeddings to our query embedding. 
# The simplest similarity measure is cosine similarity
#Embeddings: Wrapper around a text embedding model, used for converting text to embeddings
#30+ integrations to choose from
from langchain_community.vectorstores import Chroma

#CREATE INITIAL VECTORS
#reading and saving it to disk
# print(datetime.now())
# vectorstore = Chroma.from_documents(documents=all_splits, 
									# persist_directory=chroma_db_persist,
									# embedding=embedding_openai) #this actually calls openAI, most likely ada
# vectorstore.persist()
# print(datetime.now())								



#reading vectors from disk
#https://python.langchain.com/docs/integrations/vectorstores/chroma 
#and this guy to the rescue! https://www.youtube.com/watch?v=0TtwlSHo7vQ
print(datetime.now())
vectorstore = Chroma(persist_directory=chroma_db_persist, embedding_function=embedding_openai) #this actually calls openAI, most likely ada
vectorstore.get()
print(datetime.now())



# Now let’s write the actual application logic. We want to create a simple application 
# that takes a user question, searches for documents relevant to that question, 
# passes the retrieved documents and initial question to a model, and returns an answer.


# First we need to define our logic for searching over documents. LangChain defines a 
# Retriever interface which wraps an index that can return relevant Documents given a string query.
# The most common type of Retriever is the VectorStoreRetriever, which uses the similarity 
# search capabilities of a vector store to facillitate retrieval. Any VectorStore can easily be 
# turned into a Retriever with VectorStore.as_retriever()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke(our_query)
print(len(retrieved_docs))
print(retrieved_docs[0].page_content)
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


# Let’s put it all together into a chain that takes a question, retrieves relevant documents, 
# constructs a prompt, passes that to a model, and parses the output.
from langchain_openai import ChatOpenAI
from langchain import hub

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key)
#this prompt just pulls a prompt from this URL. No idea why because you can just type it, probably the devs were lazy or something, no idea
#we also dont have to use this code, we can just use normal openAI from here on out, the stuff above about Vector DBs are important
prompt = hub.pull("rlm/rag-prompt") #can be found here https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=8a81b056-3d69-597c-9690-30c1f968dd2b

example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()
example_messages

print("********************************************")
#lets customize our prompt like normal people are not use their prompt hub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

#rag_chain.invoke(our_query)

for chunk in rag_chain.stream(our_query):
	print(chunk, end="", flush=True)