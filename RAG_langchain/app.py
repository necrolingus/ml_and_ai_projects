#flask run --port 1555 --host 0.0.0.0
from flask import Flask, render_template, request
import sys, os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



openai_key = os.environ["OPENAI_CHATGPT_API_KEY"]
chroma_db_persist = 'c:/tmp/mytestChroma3/' #chroma will create the folders if they do not exist
embedding_openai = OpenAIEmbeddings(openai_api_key = openai_key)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_key)


def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)


app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')
	
	
@app.route('/hello')
def hello():
	#our_query = "Whay is Nutun so secure?"
	our_query = request.args.get('question')
		
	#return {'data':'hi there'}
	loader = TextLoader("./mytext.txt")
	docs = loader.load()

	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
	all_splits = text_splitter.split_documents(docs)
	
	
	vectorstore = Chroma.from_documents(documents=all_splits, 
										persist_directory=chroma_db_persist,
										embedding=embedding_openai) #this actually calls openAI, most likely ada
										
	retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
	retrieved_docs = retriever.invoke(our_query)
	
	print(len(retrieved_docs))
	print(retrieved_docs[0].page_content)
	
	prompt = hub.pull("rlm/rag-prompt") #can be found here https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=8a81b056-3d69-597c-9690-30c1f968dd2b
	example_messages = prompt.invoke(
		{"context": "filler context", "question": "filler question"}
	).to_messages()
	
	
	template = """Use the following pieces of context to answer the question at the end.
		If you don't know the answer, just say that you don't know, don't try to make up an answer.
		Use three sentences maximum and keep the answer as concise as possible.

		{context}

		Question: {question}

		Helpful Answer:
	"""
	custom_rag_prompt = PromptTemplate.from_template(template)
	rag_chain = (
		{"context": retriever | format_docs, "question": RunnablePassthrough()}
		| custom_rag_prompt
		| llm
		| StrOutputParser()
	)

	resp = ""
	for chunk in rag_chain.stream(our_query):
		resp = resp + chunk
		#print(chunk, end="", flush=True)
	return {'data':resp}
	
	