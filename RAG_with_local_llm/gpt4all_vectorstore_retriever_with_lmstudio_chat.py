from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

#setup variables
chroma_db_persist = 'c:/tmp/mytestChroma3_1/' #chroma will create the folders if they do not exist
our_query = "what moneytary value does nutun process?"


#setup objects
gpt4all_embd = GPT4AllEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80, add_start_index=True)


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
retrieved_docs = retriever.invoke(our_query)


#see the results
print(len(retrieved_docs))
print(retrieved_docs[0].page_content)


#Lets start building our LLM agent
#https://lmstudio.ai/docs/local-server
#https://platform.openai.com/docs/api-reference/audio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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
llm = ChatOpenAI(temperature=0.5, base_url="http://localhost:1234/v1", api_key="lmstudio")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream(our_query):
	print(chunk, end="", flush=True)
