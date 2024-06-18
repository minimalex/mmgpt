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
def pancreas_t_stage(tumor_size: float, extends_to_celiac_axis: bool, extends_to_superior_mesenteric_artery: bool) -> str:
    """
    Calculates the T stage for pancreatic cancer.
    """
    if extends_to_celiac_axis or extends_to_superior_mesenteric_artery:
        return "T4"
    elif tumor_size > 4:
        return "T3"
    elif tumor_size > 2:
        return "T2"
    elif tumor_size <= 2:
        return "T1"
    else:
        return "TX"
    
@tool
def colorectal_t_stage(tumor_size: float, invasion_depth: str) -> str:
    """
    Calculates the T stage for colorectal cancer.
    """
    if invasion_depth == "submucosa":
        return "T1"
    elif invasion_depth == "muscularis propria":
        return "T2"
    elif invasion_depth == "through muscularis propria into pericolorectal tissues":
        return "T3"
    elif invasion_depth == "through visceral peritoneum or invades other organs":
        return "T4"
    else:
        return "TX"

@tool
def pancreas_n_stage(number_of_positive_nodes: int) -> str:
    """
    Calculates the N stage for pancreatic cancer.
    """
    if number_of_positive_nodes > 3:
        return "N2"
    elif 1 <= number_of_positive_nodes <= 3:
        return "N1"
    else:
        return "N0"
    
@tool
def colorectal_n_stage(number_of_positive_nodes: int) -> str:
    """
    Calculates the N stage for colorectal cancer.
    """
    if number_of_positive_nodes == 0:
        return "N0"
    elif 1 <= number_of_positive_nodes <= 3:
        return "N1"
    elif number_of_positive_nodes >= 4:
        return "N2"
    else:
        return "NX"

@tool
def pancreas_m_stage(distant_metastasis: bool) -> str:
    """
    Calculates the M stage for pancreatic cancer.
    """
    if distant_metastasis:
        return "M1"
    else:
        return "M0"
    
@tool
def colorectal_m_stage(distant_metastasis: bool, peritoneal_metastasis: bool) -> str:
    """
    Calculates the M stage for colorectal cancer.
    """
    if distant_metastasis and peritoneal_metastasis:
        return "M1c"
    elif distant_metastasis:
        return "M1a"
    elif peritoneal_metastasis:
        return "M1b"
    else:
        return "M0"

@tool
def calculate_gps(crp: float, albumin: float) -> str:
    """
    Calculates the Glasgow Prognostic Score (GPS) for pancreatic cancer.
    """
    if crp > 10 and albumin < 3.5:
        return "GPS 2"
    elif crp > 10 or albumin < 3.5:
        return "GPS 1"
    else:
        return "GPS 0"

@tool
def colorectal_cea_levels(cea_level: float) -> str:
    """
    Checks the CEA (carcinoembryonic antigen) levels to monitor colorectal cancer.
    """
    if cea_level < 5.0:
        return "CEA level is within normal range"
    elif cea_level >= 5.0:
        return "Elevated CEA level, which may indicate active disease or progression"
    else:
        return "Invalid CEA level"

@tool
def colorectal_molecular_profile(msi_status: str, mmr_status: str) -> str:
    """
    Checks the molecular and genetic profile for colorectal cancer.
    """
    msi = msi_status.lower()
    mmr = mmr_status.lower()
    
    if msi == "msi-high" and mmr == "deficient":
        return "MSI-High and MMR-Deficient, which may indicate better prognosis and response to immunotherapy"
    elif msi == "mss" and mmr == "proficient":
        return "Microsatellite Stable and MMR-Proficient, standard prognosis"
    else:
        return "Unusual MSI/MMR combination, further analysis required"
    
@tool
def calculate_bclc_stage(tumor_size: float, number_of_nodules: int, child_pugh_class: str, performance_status: int, cancer_related_symptoms: bool) -> str:
    """
    Calculates the BCLC stage for hepatocellular carcinoma.
    """
    if child_pugh_class == "C" or performance_status > 2:
        return "Stage D (Terminal)"
    if cancer_related_symptoms or performance_status == 2:
        return "Stage C (Advanced)"
    if number_of_nodules > 3 or tumor_size > 3:
        return "Stage B (Intermediate)"
    if number_of_nodules <= 3 and tumor_size <= 3 and child_pugh_class == "A" and performance_status == 0:
        return "Stage 0 (Very early)"
    if number_of_nodules <= 3 and tumor_size <= 3 and (child_pugh_class == "A" or child_pugh_class == "B") and performance_status <= 1:
        return "Stage A (Early)"
    return "Unclassified"

@tool
def calculate_child_pugh_score(bilirubin: float, albumin: float, inr: float, ascites: str, encephalopathy: str) -> str:
    """
    Calculates the Child-Pugh score for liver disease.
    """
    score = 0

    # Bilirubin points
    if bilirubin < 2:
        score += 1
    elif bilirubin <= 3:
        score += 2
    else:
        score += 3

    # Albumin points
    if albumin > 3.5:
        score += 1
    elif albumin >= 2.8:
        score += 2
    else:
        score += 3

    # INR points
    if inr < 1.7:
        score += 1
    elif inr <= 2.3:
        score += 2
    else:
        score += 3

    # Ascites points
    if ascites == "None":
        score += 1
    elif ascites == "Mild":
        score += 2
    else:
        score += 3

    # Encephalopathy points
    if encephalopathy == "None":
        score += 1
    elif encephalopathy == "Grade 1-2":
        score += 2
    else:
        score += 3

    # Determine class
    if score <= 6:
        return "Class A (Well-compensated disease)"
    elif score <= 9:
        return "Class B (Significant functional compromise)"
    else:
        return "Class C (Decompensated disease)"

@tool
def calculate_meld_score(bilirubin: float, inr: float, creatinine: float) -> float:
    """
    Calculates the MELD score for liver disease.
    """
    import math
    meld_score = 3.78 * math.log(bilirubin) + 11.2 * math.log(inr) + 9.57 * math.log(creatinine) + 6.43
    return round(meld_score, 2)

toolkit = [retrieve_cpg, pancreas_t_stage, colorectal_t_stage, pancreas_n_stage, colorectal_n_stage, pancreas_m_stage, colorectal_m_stage, calculate_gps, colorectal_cea_levels, colorectal_molecular_profile, calculate_bclc_stage, calculate_child_pugh_score, calculate_meld_score]
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
