# import streamlit as st

# # Must be the first Streamlit command
# st.set_page_config(
#     page_title="Document Chat",
#     page_icon="🤖",
#     layout="wide"
# )

# import openai
# import os
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings  # Updated import
# from langchain_community.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from langchain_openai import ChatOpenAI  # Updated import
# from langchain_community.document_loaders import PDFPlumberLoader  # Changed to PDFPlumber
# import tempfile
# from typing import List, Tuple, Dict

# load_dotenv()

# # Configure OpenAI API key
# if "OPENAI_API_KEY" not in st.session_state:
#     st.session_state.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# class DocumentProcessor:
#     def __init__(self):
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         self.embeddings = OpenAIEmbeddings()

#     def process_file(self, file) -> Tuple[List[str], FAISS]:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#             tmp_file.write(file.getvalue())
#             tmp_file.flush()
            
#             loader = PDFPlumberLoader(tmp_file.name)
#             data = loader.load()
            
#         texts = self.text_splitter.split_documents(data)
#         vectorstore = FAISS.from_documents(texts, self.embeddings)
#         raw_text = [doc.page_content for doc in texts]
        
#         os.unlink(tmp_file.name)
#         return raw_text, vectorstore

# class ChatInterface:
#     def __init__(self):
#         self.llm = ChatOpenAI(
#             model_name="gpt-4",
#             temperature=0,
#             max_tokens=1000
#         )

#     def create_chain(self, vectorstore):
#         return ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=vectorstore.as_retriever(
#                 search_kwargs={"k": 3}
#             ),
#             return_source_documents=True,
#             verbose=False
#         )

# def initialize_session_state():
#     session_vars = ["messages", "vectorstore", "raw_text", "chat_chain", "document_summary"]
#     for var in session_vars:
#         if var not in st.session_state:
#             st.session_state[var] = [] if var == "messages" else None

# def summarize_document(texts: List[str]) -> str:
#     summary = " ".join(texts[:3])
#     return summary

# def main():
#     st.title("🤖 Document Chat")
    
#     # Add API key input to sidebar if not set
#     if not st.session_state.OPENAI_API_KEY:
#         with st.sidebar:
#             st.session_state.OPENAI_API_KEY = st.text_input("Enter OpenAI API key:", type="password")
#             if not st.session_state.OPENAI_API_KEY:
#                 st.warning("Please enter your OpenAI API key to continue.")
#                 st.stop()

#     # Set the API key
#     openai.api_key = st.session_state.OPENAI_API_KEY
    
#     initialize_session_state()
#     doc_processor = DocumentProcessor()
#     chat_interface = ChatInterface()
    
#     with st.sidebar:
#         st.header("📁 Document Upload")
#         uploaded_file = st.file_uploader(
#             "Upload your PDF",
#             type=["pdf"],
#             help="Upload a PDF document to chat with"
#         )
        
#         if uploaded_file and "processed_file" not in st.session_state:
#             with st.spinner("Processing document..."):
#                 raw_text, vectorstore = doc_processor.process_file(uploaded_file)
#                 st.session_state.raw_text = raw_text
#                 st.session_state.vectorstore = vectorstore
#                 st.session_state.chat_chain = chat_interface.create_chain(vectorstore)
#                 st.session_state.processed_file = True
#                 st.session_state.messages = []
#                 st.session_state.document_summary = summarize_document(raw_text)
#             st.success("Document processed successfully!")
            
#         if st.button("Clear Chat History"):
#             st.session_state.messages = []
            
#     if "processed_file" in st.session_state:
#         with st.expander("Document Summary"):
#             st.write(st.session_state.document_summary)
        
#         for msg in st.session_state.messages:
#             with st.chat_message(msg["role"]):
#                 st.write(msg["content"])
        
#         if prompt := st.chat_input("Ask about your document"):
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.write(prompt)
            
#             with st.chat_message("assistant"):
#                 with st.spinner("Thinking..."):
#                     result = st.session_state.chat_chain({
#                         "question": prompt,
#                         "chat_history": [(m["role"], m["content"]) 
#                                        for m in st.session_state.messages[:-1]]
#                     })
#                     response = result["answer"]
#                     st.write(response)
                    
#                     if "source_documents" in result:
#                         with st.expander("View Sources"):
#                             for i, doc in enumerate(result["source_documents"]):
#                                 st.markdown(f"**Source {i+1}:**")
#                                 st.markdown(doc.page_content)
#                                 st.markdown("---")
            
#             st.session_state.messages.append({"role": "assistant", "content": response})
#     else:
#         st.info("Please upload a document to start chatting!")

# if __name__ == "__main__":
#     main()


# import streamlit as st
# import openai
# import os
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from langchain_openai import ChatOpenAI
# from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader, TextLoader
# import tempfile
# from typing import List, Tuple, Dict

# # Load environment variables
# load_dotenv()

# # Configure Streamlit page
# st.set_page_config(
#     page_title="Advanced Document Chat",
#     page_icon="🤖",
#     layout="wide"
# )

# # Configure OpenAI API key
# if "OPENAI_API_KEY" not in st.session_state:
#     st.session_state.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# class DocumentProcessor:
#     def __init__(self):
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         self.embeddings = OpenAIEmbeddings()

#     def process_file(self, file) -> Tuple[List[str], FAISS]:
#         """Process uploaded file and return raw text and vectorstore."""
#         with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
#             tmp_file.write(file.getvalue())
#             tmp_file.flush()
            
#             # Load document based on file type
#             if file.name.endswith('.pdf'):
#                 loader = PDFPlumberLoader(tmp_file.name)
#             elif file.name.endswith('.docx'):
#                 loader = Docx2txtLoader(tmp_file.name)
#             elif file.name.endswith('.txt'):
#                 loader = TextLoader(tmp_file.name)
#             else:
#                 raise ValueError("Unsupported file format")
            
#             data = loader.load()
            
#         # Split and embed text
#         texts = self.text_splitter.split_documents(data)
#         vectorstore = FAISS.from_documents(texts, self.embeddings)
#         raw_text = [doc.page_content for doc in texts]
        
#         # Clean up temporary file
#         os.unlink(tmp_file.name)
#         return raw_text, vectorstore

# class ChatInterface:
#     def __init__(self):
#         self.llm = ChatOpenAI(
#             model_name="gpt-4",
#             temperature=0,
#             max_tokens=1000
#         )

#     def create_chain(self, vectorstore):
#         """Create a conversational retrieval chain."""
#         return ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=vectorstore.as_retriever(
#                 search_kwargs={"k": 5}  # Retrieve more documents for better context
#             ),
#             return_source_documents=True,
#             verbose=False
#         )

# def initialize_session_state():
#     """Initialize session state variables."""
#     session_vars = {
#         "messages": [],
#         "vectorstore": None,
#         "raw_text": None,
#         "chat_chain": None,
#         "document_summary": None,
#         "processed_file": False
#     }
#     for var, default_value in session_vars.items():
#         if var not in st.session_state:
#             st.session_state[var] = default_value

# def summarize_document(texts: List[str]) -> str:
#     """Generate a summary of the document."""
#     summary = " ".join(texts[:3])  # Use the first 3 chunks for summary
#     return summary

# def main():
#     st.title("🤖 Advanced Document Chat")
    
#     # Add API key input to sidebar if not set
#     if not st.session_state.OPENAI_API_KEY:
#         with st.sidebar:
#             st.session_state.OPENAI_API_KEY = st.text_input("Enter OpenAI API key:", type="password")
#             if not st.session_state.OPENAI_API_KEY:
#                 st.warning("Please enter your OpenAI API key to continue.")
#                 st.stop()

#     # Set the API key
#     openai.api_key = st.session_state.OPENAI_API_KEY
    
#     # Initialize session state
#     initialize_session_state()
#     doc_processor = DocumentProcessor()
#     chat_interface = ChatInterface()
    
#     # Sidebar for document upload and settings
#     with st.sidebar:
#         st.header("📁 Document Upload")
#         uploaded_file = st.file_uploader(
#             "Upload your document",
#             type=["pdf", "docx", "txt"],
#             help="Upload a document to chat with"
#         )
        
#         if uploaded_file and not st.session_state.processed_file:
#             with st.spinner("Processing document..."):
#                 try:
#                     raw_text, vectorstore = doc_processor.process_file(uploaded_file)
#                     st.session_state.raw_text = raw_text
#                     st.session_state.vectorstore = vectorstore
#                     st.session_state.chat_chain = chat_interface.create_chain(vectorstore)
#                     st.session_state.processed_file = True
#                     st.session_state.messages = []
#                     st.session_state.document_summary = summarize_document(raw_text)
#                     st.success("Document processed successfully!")
#                 except Exception as e:
#                     st.error(f"Error processing document: {e}")
            
#         if st.button("Clear Chat History"):
#             st.session_state.messages = []
            
#     # Main chat interface
#     if st.session_state.processed_file:
#         # Display document summary
#         with st.expander("Document Summary"):
#             st.write(st.session_state.document_summary)
        
#         # Display chat history
#         for msg in st.session_state.messages:
#             with st.chat_message(msg["role"]):
#                 st.write(msg["content"])
        
#         # Handle user input
#         if prompt := st.chat_input("Ask about your document"):
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.write(prompt)
            
#             with st.chat_message("assistant"):
#                 with st.spinner("Thinking..."):
#                     try:
#                         if st.session_state.chat_chain is None:
#                             st.error("Chat chain not initialized. Please upload a document.")
#                         else:
#                             # Retrieve and generate response
#                             result = st.session_state.chat_chain({
#                                 "question": prompt,
#                                 "chat_history": [(m["role"], m["content"]) 
#                                                for m in st.session_state.messages[:-1]]
#                             })
#                             response = result["answer"]
#                             st.write(response)
                            
#                             # Display source documents
#                             if "source_documents" in result:
#                                 with st.expander("View Sources"):
#                                     for i, doc in enumerate(result["source_documents"]):
#                                         st.markdown(f"**Source {i+1}:**")
#                                         st.markdown(doc.page_content)
#                                         st.markdown("---")
                            
#                             # Append assistant's response to chat history
#                             st.session_state.messages.append({"role": "assistant", "content": response})
#                     except Exception as e:
#                         st.error(f"Error generating response: {e}")
#     else:
#         st.info("Please upload a document to start chatting!")

# if __name__ == "__main__":
#     main()




import streamlit as st
import openai
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader, TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import tempfile
from typing import List, Tuple, Dict
import nltk
from nltk.tokenize import sent_tokenize
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🤖",
    layout="wide"
)

class DocumentProcessor:
    def __init__(self):
        # Enhanced text splitting with better parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for more precise retrieval
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],  # More granular splitting
            is_separator_regex=False
        )
        self.embeddings = OpenAIEmbeddings()

    def preprocess_text(self, text: str) -> str:
        """Apply text preprocessing steps."""
        try:
            # Remove extra whitespace
            text = " ".join(text.split())
            # Basic sentence splitting if NLTK fails
            if not text:
                return text
            try:
                # Try NLTK sentence tokenization
                sentences = sent_tokenize(text)
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}. Using basic splitting.")
                # Fallback to basic sentence splitting
                sentences = [s.strip() for s in text.split('.') if s.strip()]
            return " ".join(sentences)
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            return text  # Return original text if processing fails

    def process_file(self, file) -> Tuple[List[str], FAISS]:
        """Process uploaded file with enhanced error handling and preprocessing."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file.flush()
                
                # Enhanced loader selection
                loaders = {
                    '.pdf': PDFPlumberLoader,
                    '.docx': Docx2txtLoader,
                    '.txt': TextLoader
                }
                
                file_ext = os.path.splitext(file.name)[1].lower()
                if file_ext not in loaders:
                    raise ValueError(f"Unsupported file format: {file_ext}")
                
                loader = loaders[file_ext](tmp_file.name)
                data = loader.load()
                
                # Preprocess and split text
                processed_docs = []
                for doc in data:
                    doc.page_content = self.preprocess_text(doc.page_content)
                    processed_docs.append(doc)
                
                texts = self.text_splitter.split_documents(processed_docs)
                
                # Create FAISS index
                # Add source metadata to each document
                for text in texts:
                    text.metadata["source"] = file.name
                    
                vectorstore = FAISS.from_documents(
                    texts,
                    self.embeddings
                )
                
                raw_text = [doc.page_content for doc in texts]
                
                os.unlink(tmp_file.name)
                return raw_text, vectorstore
                
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

class AdvancedChatInterface:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0.3,  # Slightly increased for more natural responses
            max_tokens=2000
        )
        
        # Custom prompt template for better context handling
        self.qa_template = """You are a helpful AI assistant with expertise in analyzing documents. 
        Use the following context to answer the user's question. If you're unsure or the context 
        doesn't contain the relevant information, say so.

        Context: {context}
        
        Chat History: {chat_history}
        
        Question: {question}
        
        Answer: Let me help you with that."""
        
        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "chat_history", "question"]
        )

    def create_chain(self, vectorstore):
        """Create an enhanced conversational retrieval chain."""
        # Create memory with summary
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Create LLM-based document compressor
        compressor = LLMChainExtractor.from_llm(self.llm)
        
        # Create compressed retriever
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            base_compressor=compressor
        )
        
        # Create the chain with enhanced configuration
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=compression_retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            verbose=True
        )

def generate_detailed_summary(texts: List[str], llm: ChatOpenAI) -> str:
    """Generate a more detailed document summary using GPT-4."""
    summary_template = """
    Please provide a comprehensive summary of the following document. 
    Include key topics, main points, and any notable findings or conclusions.
    
    Document text:
    {text}
    
    Summary:"""
    
    summary_prompt = PromptTemplate(template=summary_template, input_variables=["text"])
    
    # Combine first few chunks for summary
    text_for_summary = " ".join(texts[:5])
    
    try:
        response = llm.predict(summary_prompt.format(text=text_for_summary))
        return response
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return "Error generating summary. Using default summary instead: " + " ".join(texts[:2])

def initialize_session_state():
    """Initialize session state variables."""
    session_vars = {
        "messages": [],
        "vectorstore": None,
        "raw_text": None,
        "chat_chain": None,
        "document_summary": None,
        "processed_file": False,
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def main():
    st.title("🤖 Advanced RAG System")
    
    # Initialize components
    initialize_session_state()
    doc_processor = DocumentProcessor()
    chat_interface = AdvancedChatInterface()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload your document",
            type=["pdf", "docx", "txt"],
            help="Upload a document to chat with"
        )
        
        # Add system message configuration
        st.header("⚙️ Chat Settings")
        system_message = st.text_area(
            "System Message",
            value="You are a helpful AI assistant specialized in document analysis.",
            help="Customize the AI's behavior"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            help="Higher values make responses more creative"
        )
        
        if uploaded_file and not st.session_state.processed_file:
            with st.spinner("Processing document..."):
                try:
                    raw_text, vectorstore = doc_processor.process_file(uploaded_file)
                    st.session_state.raw_text = raw_text
                    st.session_state.vectorstore = vectorstore
                    
                    # Generate detailed summary
                    st.session_state.document_summary = generate_detailed_summary(
                        raw_text,
                        chat_interface.llm
                    )
                    
                    # Create chat chain with custom settings
                    chat_interface.llm.temperature = temperature
                    st.session_state.chat_chain = chat_interface.create_chain(vectorstore)
                    
                    st.session_state.processed_file = True
                    st.session_state.messages = []
                    st.success("Document processed successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    logger.error(f"Document processing error: {str(e)}")
    
    # Main chat interface
    if st.session_state.processed_file:
        # Display enhanced document summary
        with st.expander("📄 Document Summary", expanded=True):
            st.markdown(st.session_state.document_summary)
            
        # Chat interface with error handling
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        if prompt := st.chat_input("Ask about your document"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document and generating response..."):
                    try:
                        result = st.session_state.chat_chain({
                            "question": prompt,
                            "chat_history": [(m["role"], m["content"]) 
                                           for m in st.session_state.messages[:-1]]
                        })
                        
                        response = result["answer"]
                        st.markdown(response)
                        
                        # Enhanced source document display
                        if "source_documents" in result:
                            with st.expander("📚 View Sources"):
                                for i, doc in enumerate(result["source_documents"]):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.markdown(doc.page_content)
                                    if hasattr(doc.metadata, 'page'):
                                        st.markdown(f"*Page: {doc.metadata['page']}*")
                                    st.markdown("---")
                        
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        logger.error(f"Response generation error: {str(e)}")
    else:
        st.info("👋 Please upload a document to start chatting!")

if __name__ == "__main__":
    main()