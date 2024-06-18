import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.prompts import MessagesPlaceholder
from pinecone import Pinecone

# Initialize Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "mm-gpt"

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Load the existing Pinecone index
index = pc.Index(index_name)
docsearch = PineconeVectorStore(index=index, embedding=embeddings)
retriever = docsearch.as_retriever()

def retrieve_with_source(query):
    results = docsearch.similarity_search_with_score(query)
    contexts_with_sources = [
        {"text": result[0].page_content, "source": result[0].metadata.get("source", "unknown")}
        for result in results
    ]
    return contexts_with_sources

@tool
def retrieve_cpg(query: str) -> str:
    """
    Search and return information about diagnosis and therapeutic approaches based on the clinical practice guidelines, including the source of the information.
    """
    contexts_with_sources = retrieve_with_source(query)
    formatted_contexts = "\n\n".join(
        f"Context: {item['text']}\nSource: {item['source']}" for item in contexts_with_sources
    )
    return formatted_contexts

@tool
def calculate_iss(serum_beta2_microglobulin: float, serum_albumin: float) -> str:
    """
    Calculates the ISS score for multiple myeloma.
    """
    if serum_beta2_microglobulin < 3.5 and serum_albumin >= 3.5:
        return "Stage I"
    elif serum_beta2_microglobulin >= 5.5:
        return "Stage III"
    else:
        return "Stage II"

@tool
def calculate_r_iss(serum_beta2_microglobulin: float, serum_albumin: float, ldh: float, high_risk_cytogenetics: bool) -> str:
    """
    Calculates the R-ISS score for multiple myeloma.
    """
    if serum_beta2_microglobulin < 3.5 and serum_albumin >= 3.5 and ldh < 240 and not high_risk_cytogenetics:
        return "Stage I"
    elif serum_beta2_microglobulin >= 5.5 or ldh >= 240 or high_risk_cytogenetics:
        return "Stage III"
    else:
        return "Stage II"

toolkit = [calculate_iss, calculate_r_iss, retrieve_cpg]
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
          You are a medical assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you encounter diagnostic values in the input, you can use the tools to calculate relevant clinical scores beforehand. Try to calculate every score that matches the given values in the input. Use all appropriate tools to provide the most accurate answer. 
          """),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_openai_tools_agent(llm, toolkit, prompt)
agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=False)

# Streamlit web application
st.title("Evidence-based Bot")
st.write("Enter your medical question:")

input_query = st.text_input("Clinical Query", "")
if input_query:
    with st.spinner("Processing..."):
        result = agent_executor.invoke({"input": input_query})
        st.markdown("### I would suggest the following")
        st.markdown(result['output'])
