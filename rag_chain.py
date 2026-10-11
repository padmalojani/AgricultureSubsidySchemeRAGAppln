from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import config
from prompts import QA_PROMPT

def get_rag_chain():
    chat_model = ChatOpenAI(
        model=config.CHAT_MODEL, 
        temperature=0.1, 
        openai_api_key=config.OPENAI_API_KEY
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    return chat_model, format_docs