
def get_retrievers(pdf_path):
    import warnings
    warnings.filterwarnings("ignore")
    import random
    import pdfplumber
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from langchain.embeddings import HuggingFaceBgeEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.docstore.document import Document
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain.storage import InMemoryStore
    # from tqdm.autonotebook import tqdm, trange
    
    embedding_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en",
                               encode_kwargs={'normalize_embeddings': False})
    
    def embed_texts(texts):
        # return FastEmbedEmbeddings.embed_documents(embedding_model,texts = texts)
        return embedding_model.embed_documents(texts)
    
    def get_header_footer(pdf_path, threshold=0.71):
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages >= 15:
                random_page_nos = random.sample(range(5, total_pages), 10)
            else:
                random_page_nos = list(range(total_pages))
            
            avg_similarity = 1
            header_lines = -1
            
            while avg_similarity > threshold and header_lines < 4:
                header_lines += 1
                five_lines = []
            
                for page_no in random_page_nos:
                    lines = pdf.pages[page_no].extract_text().split('\n')
                    if len(lines) > header_lines:
                        five_lines.append(lines[header_lines])
                similarities = cosine_similarity(embed_texts(five_lines))
                avg_similarity = np.mean(similarities[np.triu_indices(len(similarities), k=1)])
                
            avg_similarity = 1
            footer_lines = -1
            
            while avg_similarity > threshold and footer_lines < 4:
                footer_lines += 1
                five_lines = []
                
                for page_no in random_page_nos:
                    lines = pdf.pages[page_no].extract_text().split('\n')
                    if len(lines) > footer_lines:
                        five_lines.append(lines[-(footer_lines+1)])
                similarities = cosine_similarity(embed_texts(five_lines))
                avg_similarity = np.mean(similarities[np.triu_indices(len(similarities), k=1)]) 
            return header_lines, footer_lines
        
    def extract_text(pdf_path):
        header_lines, footer_lines = get_header_footer(pdf_path)
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    lines = page_text.split('\n')
                    if lines:
                        page_text = '\n'.join(lines[header_lines:-(footer_lines+1)])
                        text += page_text + '\n'
            return text
    text = extract_text(pdf_path)
        
    def get_vectorstore1(path):
        texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
        docs = [Document(text) for text in texts if text.strip()]
        vectorstore = FAISS.from_documents(docs, embedding_model)
        return vectorstore
    
    def get_vectorstore2(path):
        texts = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=400).split_text(text)
        docs = [Document(text) for text in texts if text.strip()]
        vectorstore = FAISS.from_documents(docs, embedding_model)
        return vectorstore
    
    retriever1 = get_vectorstore1(pdf_path).as_retriever(search_kwargs={"k": 6})
    retriever2 = get_vectorstore2(pdf_path).as_retriever(search_kwargs={"k": 3})
    return retriever1, retriever2

# This should be inside app code as this is one time process
# retriever1, retriever2 = get_two_retrievers(pdf_path1)
# retriever3, retriever4 = get_two_retrievers(pdf_path2)
    
def get_answer(query,  retriever1, retriever2):
    import warnings
    import streamlit as st
    warnings.filterwarnings("ignore")
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain_groq import ChatGroq

    llm = ChatGroq(groq_api_key = st.secrets["API_KEY"]["GROQ"], model = 'llama3-70b-8192', temperature=0.05)
    
    qa_template = """
    Your task is to answer the question, using only the information provided in the given context.
    The answer should be accuate and detailed.
    Where applicable, refer to specific section numbers within the context (e.g., "According to section 4.1.2,...").
    If the answer is not found in the provided context, simply state that there is no relevant information available without sharing details about the context.
    
    CONTEXT: {context}
    
    QUESTION: {question}
    """
    qa_prompt = PromptTemplate(template=qa_template, input_variables=["question", "context"])
    
    combine_template = """
    Your task is to answer the question, by synthesizing relevant information from the provided answers.
    The answer should be accurate and detailed.
    Where applicable, refer to specific section numbers within the context (e.g., "According to section 4.1.2,...").
    Do not reveal that the information comes from multiple answers, directly answer the question.
    
    QUESTION: {question}
    
    ANSWER 1: {answer1}
    
    ANSWER 2: {answer2}
    """

    chain1 = RetrievalQA.from_llm(llm=llm, retriever=retriever1, prompt= qa_prompt)
    chain2 = RetrievalQA.from_llm(llm=llm, retriever=retriever2, prompt= qa_prompt)
    
    answer1 = chain1.invoke(query)['result']
    answer2 = chain2.invoke(query)['result']
    
    combine_prompt = combine_template.format(question = query,answer1 = answer1, answer2 = answer2)

    answer = llm.invoke(combine_prompt).content
    
    return answer

def compare_answers(query, retriever1, retriever2, retriever3, retriever4):
    from langchain_groq import ChatGroq
    import streamlit as st
    llm = ChatGroq(groq_api_key =st.secrets["API_KEY"]["GROQ"], model = 'llama3-70b-8192', temperature=0.05)
    
    answer1 = get_answer(query, retriever1, retriever2)
    answer2 = get_answer(query, retriever3, retriever4)
    comparison_template = """
    We have provided a question and their two answers generated from two different documents. Generate a comparison section without a heading which includes whether both answer are same or partially same or different. If they are paritially same, then what is same and what is different. This comparison is based on the answers generated from both the contexts. Accuracy and precision are crucial for this task.
    
    QUESTION: {question}
    
    ANSWER 1: {answer1}
    
    ANSWER 2: {answer2}
    """
    comparison_prompt = comparison_template.format(question = query,answer1 = answer1, answer2 = answer2)
    comparison = llm.invoke(comparison_prompt).content

    final_prompt = """
    ANSWER 1: {answer1}
    
    ANSWER 2: {answer2}

    COMPARISON:{comparison}
    """
    return final_prompt.format(answer1 = answer1, answer2 = answer2, comparison = comparison)
