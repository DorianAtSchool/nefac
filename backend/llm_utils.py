import faiss
from langchain_community.embeddings import OpenAIEmbeddings  # Replace with Bedrock embeddings if using AWS Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI  # Replace with Claude integration
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
import glob
from langchain_core.prompts import PromptTemplate
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import StuffDocumentsChain ,LLMChain
import json
import yt_dlp

import logging
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# loaders
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat

from dotenv import load_dotenv

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Initialize embeddings and FAISS vector store
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
# Initialize docstore -- MAKE PERMANENT FILE
docstore = InMemoryDocstore({})
vector_store = FAISS(embedding_function=embedding_model, index = faiss.IndexFlatL2(1536), docstore = docstore, index_to_docstore_id={})  # 768 is the embedding dimension size
all_docs = []

# Add documents to the FAISS vector store
def add_documents_to_store(_, info, documents):
    # convert doc paths to Doc objects
    #new_docs = load_documents_from_directory()
    new_docs = pdfLoader()
    new_vids = youtubeLoader()

    # Set metadata for videos (if not already set in youtubeLoader)
    for vid in new_vids:
        vid.metadata['page'] = vid.metadata.get('page', 0)
        # 'title' and 'source' are already set in youtubeLoader

    chunked_docs = chunk_documents(new_docs + new_vids)
    vector_store.add_documents(documents=chunked_docs)
    logger.info(f"Added {len(chunked_docs)} documents to the vector store.")
    return chunked_docs


# TODO: NEEDS IMPLEMENTATION
def load_documents_from_directory():
    # Load all new documents from the directory "docs/" of all types
    pass

def retrieve_documents(query):
    pass

# Function to chunk documents
def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = text_splitter.split_documents(docs)
    return chunked_docs

# Function to ask the LLM
async def ask_llm(_, info, query):

    response = await custom_QA(_, info, query)
    
    return response['results']
    


async def custom_QA(_, info, query):

    prompt_template = """

    Use the following context to answer the query.

    Sources:
    {context}

    Instructions:
    - Generate a list of unique relevant sources from the context.
    - Provide the actual titles and links of the sources.
    - Summarize each source content in a way that answers the query.
    - Do not include duplicate sources or sources that are not relevant to the query.
    - Format the output as JSON in the following structure:
    {{
        "results": [
            {{
                "title": "Title of the source content",
                "link": "source_link",
                "summary": " Details answering to the query with full context (who what when why where how)",
                "citations": [
                    {{"id": "1", "context": "relevant quote/text use in summary"}},
                ]
            }},
            ...
        ]
    }}

    Question: {question}
    """

    retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20},
    )

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=OpenAI(temperature = 0), prompt=QA_CHAIN_PROMPT, callbacks=None, verbose=True)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source", "page"],
        template="Context:\ncontent:{page_content}\nsource:{source}\npage:{page}\n",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None,
    )
    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        callbacks=None,
        verbose=True,
        retriever=retriever,
        return_source_documents = True,
    )
    response = qa(query)
    print(response)

    # Parse the LLM's JSON response
    try:
        result = json.loads(response['result'])
        return result
    except Exception as e:
        logger.error(f"Error parsing the LLM response: {e}")
        # Fallback logic
        formatted_response = {
            "results": []
        }
        unique_titles = set()
        for i, doc in enumerate(response.get('source_documents', [])):
            title = doc.metadata.get("title", f"Source {i+1}")
            if title in unique_titles:
                continue
            unique_titles.add(title)

            # Generate a summary relevant to the query
            llm = OpenAI(temperature=0)
            summary_prompt = f"Based on the following content, provide a concise summary that answers the query: '{query}'\n\nContent:\n{doc.page_content}"
            try:
                summary = llm(summary_prompt).strip()
            except Exception as summ_err:
                logger.error(f"Error generating summary: {summ_err}")
                summary = "Summary unavailable."

            formatted_response["results"].append({
                "title": title,
                "link": doc.metadata.get("source", "#"),
                "summary": summary,
                "citations": [
                    {
                        "id": str(i+1),
                        "context": doc.page_content
                    }
                ]
            })
        return formatted_response
        

def pdfLoader():
    # Load all PDF documents from the directory "docs/"
    all_docs_path = "docs/legal_docs/*.pdf"  # Ensure you're targeting PDFs
    all_doc_paths = glob.glob(all_docs_path)
    all_pages = []
    print(all_doc_paths)
    for idx, doc_path in enumerate(all_doc_paths):
        loader = PyPDFLoader(doc_path)
        pages = loader.load_and_split()

        # Attempt to extract the title from the PDF metadata
        if pages and pages[0].metadata:
            pdf_metadata = pages[0].metadata
            title = pdf_metadata.get('pdf:Title')
            if not title:
                # If title metadata is missing, use the file name without extension
                title = os.path.basename(doc_path).replace('.pdf', '')
        else:
            title = os.path.basename(doc_path).replace('.pdf', '')

        # Assign the title and source to each page's metadata
        for page in pages:
            page.metadata['title'] = title
            page.metadata['source'] = doc_path  # Or use a URL if available
        all_pages.extend(pages)

    return all_pages

def youtubeLoader():
    

    logger = logging.getLogger(__name__)
    youtube_url = "https://www.youtube.com/watch?v=0oMe5k7MPxs"
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'forcejson': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            video_title = info_dict.get('title', 'Video Title')
            video_url = info_dict.get('webpage_url', youtube_url)
            video_description = info_dict.get('description', '')

            # Using youtube_transcript_api to get transcript
            from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
            try:
                transcript = YouTubeTranscriptApi.get_transcript(info_dict.get('id'))
                transcript_text = " ".join([entry['text'] for entry in transcript])
            except TranscriptsDisabled:
                logger.warning(f"Transcripts are disabled for video {youtube_url}.")
                transcript_text = ""

            documents = []
            if transcript_text:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(transcript_text)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'title': video_title,
                            'source': video_url,
                            'page': i
                        }
                    )
                    documents.append(doc)
            else:
                logger.warning(f"No transcript available for video {youtube_url}.")
                # Create a Document with the description if transcript is unavailable
                doc = Document(
                    page_content=video_description,
                    metadata={
                        'title': video_title,
                        'source': video_url,
                        'page': 0
                    }
                )
                documents.append(doc)

            return documents

    except Exception as e:
        logger.error(f"Error loading YouTube video: {e}")
        return []

add_documents_to_store(" ", "info", " ")