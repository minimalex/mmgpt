from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

pc = Pinecone(api_key=pinecone_api_key)

urls = [
    "https://www.aerzteblatt.de/archiv/154415/Diagnose-und-Therapieoptionen-beim-hepatozellulaeren-Karzinom",
    "https://emedicine.medscape.com/article/280605-guidelines?form=fpf",
    "https://www.sciencedirect.com/science/article/pii/S0923753423008244?via%3Dihub",
    "https://www.annalsofoncology.org/article/S0923-7534(22)04192-8/fulltext",
    "https://www.sciencedirect.com/science/article/pii/S0923753422041928?via%3Dihub",
    "https://ascopubs.org/doi/full/10.1200/JCO.22.01690",
    "https://www.onkopedia.com/de/onkopedia/guidelines/kolonkarzinom/@@guideline/html/index.html",
    "https://www.annalsofoncology.org/article/S0923-7534(23)00824-4/fulltext"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
index_name = "mm-gpt"
index = pc.Index(index_name)
docsearch = PineconeVectorStore.from_documents(doc_splits, embeddings, index_name=index_name)
