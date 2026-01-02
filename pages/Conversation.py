import streamlit as st
import torch
from huggingface_hub import login
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

login("")   # hugging face login id 

# LLM Model 
@st.cache_resource(show_spinner="Loading model..")
def llm_model():
    llm = HuggingFacePipeline.from_model_id(
                model_id="google/gemma-2-2b-it",
                task="text-generation",
                pipeline_kwargs={"max_new_tokens": 500},
                device = -1 
            )

    return ChatHuggingFace(llm = llm)

model = llm_model()

# Embedding Model 
@st.cache_resource(show_spinner="Loading model..") 
def embedding_model():
    return HuggingFaceEmbeddings(model_name = "google/embeddinggemma-300m")

embd_model = embedding_model()

# Creating Prompt Template 
prompt = PromptTemplate(
    template = """
You are a retrieval-augmented assistant.

Use ONLY the information provided in the CONTEXT section to answer the QUESTION.
Do NOT use prior knowledge, assumptions, or external information.
If the answer is not explicitly stated in the CONTEXT, respond with:

"I do not know based on the provided context."

Do not rephrase the question.
Do not add explanations beyond what is present in the CONTEXT.
Do not speculate.

CONTEXT: {context}

QUESTION: {question}
""",
input_variables=['context', 'question']
)

st.title("Chat with YouTube Video")

if "video_url" not in st.session_state:
    st.error("No video URL found.")
    st.stop()

video_url = st.session_state["video_url"]

# BUILDING THE VIDEO PIPELINE 
@st.cache_resource(show_spinner="Processing video ..")
def video_retriever(video_url):
    # loading Youtube video Transcripts
    try:
        loader = YoutubeLoader.from_youtube_url(
            video_url,
            add_video_info = False,
            language = ["en"]
        )

        transcript_document = loader.load()
    except:
        st.error("Failed to load video transcripts !!")

    # Splitting the transcripts into chunks 
    splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 150)
    transcript_chunks = splitter.split_documents(transcript_document)

    # generating and storing embeddings in a vector database 
    vector_store = Chroma.from_documents(transcript_chunks, embd_model) 

    # Creating a reteriver 
    ret = vector_store.as_retriever(search_type = "mmr", search_kwargs = {"k" : 4})

    return ret

# extracting the related text for the retrieved documents
def page_content_extractor(documents):
    text = []
    for doc in documents:
        text.append(doc.page_content)
    combined_text = " ".join(text)
    return combined_text

# creating chains
context = RunnableLambda(page_content_extractor) 
retriever = video_retriever(video_url)

parallel_chain = RunnableParallel({
    "context" : retriever | context,
    "question" : RunnablePassthrough()
}) 

main_chain = parallel_chain | prompt | model

# user_input
user_query = st.chat_input("Ask a question")

if user_query:
    outputs = main_chain.invoke(user_query).content
    AI_response = outputs.split("<start_of_turn>model")[-1]
    st.chat_message("user").write(user_query)
    st.chat_message("assistant").write(AI_response)
