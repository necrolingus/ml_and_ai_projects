import sys, os
import chromadb
from chromadb.utils import embedding_functions


#setup variables
chroma_db_persist = 'c:/tmp/mytestChroma3/' #chroma will create the folders if they do not exist
chroma_collection_name = "my_lmstudio_test"
embed_model = "all-MiniLM-L6-v2"
large_document_list = []
large_embedding_list = []
large_metadata_list = [] 
large_ids_list = [] 


#setup objects
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
chroma_client = chromadb.PersistentClient(path=chroma_db_persist)
chroma_collection = chroma_client.get_or_create_collection(name=chroma_collection_name, embedding_function=embedding_func)

#read our file
#I split the data on ***
with open("mytext2.txt", "r") as mytext:
	content = mytext.read()
	content_list = content.split('***')
	
chroma_collection.add(
	documents = content_list,
	ids = [f"id{i}" for i in range(len(content_list))]
	#metadatas = [list of dicts] not needed
)

query_results = chroma_collection.query(query_texts=["how much do we anually process"], n_results=3)
print(query_results.keys())
print(query_results["documents"])
print(query_results["ids"])
print(query_results["distances"])
print()
sys.exit()


