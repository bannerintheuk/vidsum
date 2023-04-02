import streamlit as st
import sys
sys.tracebacklimit = 0

from langchain.llms import OpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="VidSum", page_icon=None, layout="centered", menu_items=None)
st.title("VidSum")
st.write("""
Discover VidSum, the cutting-edge video summary app that combines the power of ChatGPT and LangChain to deliver precise and succinct summaries of your favorite YouTube videos. Revolutionize your video-watching experience by quickly grasping essential insights and saving time on lengthy content.
""")
st.write("""
Discover VidSum, the cutting-edge video summary app that combines the power of ChatGPT and LangChain to deliver precise and succinct summaries of your favorite YouTube videos. Revolutionize your video-watching experience by quickly grasping essential insights and saving time on lengthy content.

**Key Features:**

* AI-Driven Summaries: VidSum employs ChatGPT, the state-of-the-art language model by OpenAI, to comprehend and condense video transcripts into clear, digestible summaries.

* Multilingual Support: VidSum offers translation and summarization capabilities in multiple languages, allowing users to access content from around the globe.

* Time-Saving: Efficiently navigate through content by concentrating on key points, enabling you to absorb more information in less time.

* Customizable Length: Select the desired length of summaries to suit your needs.
""")

st.write("VidSum Requires a Valid OpenAI API Key")
st.write("[Goto OpenAI](https://www.openai.com/)")
OPENAI_API_KEY = st.text_input("**OpenAI API Key**")


video_url = st.text_input("**Youtube Video URL**")
summary_type_options = ['Concise', 'Expanded', 'Full','Key Facts']
language_options = ['English UK', 'English US', 'Spanish','French',"Mandarin","Japanese"]

col_1,col_2 = st.columns(2)

with col_1:
    summary_type = st.selectbox('**Summary Type:**', summary_type_options)
with col_2:
    language = st.selectbox('**Language:**', language_options)


summarize = st.button("Summarize")

def create_prompt_template(summary_type:str, language:str,content:str)->str:
    template = f"Write a {summary_type} summary of the following:\n\n{content}\n\n{summary_type.upper()} SUMMARY IN {language.upper()}"
    return template


def summarize_transcript(OPENAI_API_KEY:str,summary_type,language:str):
    
    loader = YoutubeLoader.from_youtube_channel(video_url, add_video_info=True)
    result = loader.load()
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=1000)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_documents(result)

    PROMPT = PromptTemplate(template=create_prompt_template(summary_type,language,"{text}"), input_variables=["text"])
    
    
    chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)
    chain.run(chunks)
    output = chain.run(chunks)
    col_1,col_2 = st.columns(2)
    with col_1:
        st.write("## Transcript Summary")
        st.write(f"**Title**:")
        st.write(f"{result[0].metadata['title']}")
        st.write(f"**Author**:")
        st.write(f"{result[0].metadata['author']}")
        duration = result[0].metadata['length'] / 60
        st.write("**Length**:")
        st.write("{:.2f} mins".format(duration))
    with col_2:
        st.image(
                result[0].metadata["thumbnail_url"],
                width=400, # Manually Adjust the width of the image as per requirement
            )
    st.write("### Summarized Content")
    st.write(output)

if OPENAI_API_KEY and video_url and summarize:
    try:
        with st.spinner("Loading..."):
            summarize_transcript(OPENAI_API_KEY,summary_type,language)
    except ValueError as e:
        st.error(e.message, icon="🚨")
        