
import streamlit as st
from langchain_groq import ChatGroq
import tempfile
from utils import get_retrievers, get_answer, compare_answers

def main():
    st.set_page_config(page_title="Chatbot", page_icon="💬")

    # Sidebar - Upload PDF
    option = st.sidebar.selectbox('Select an option', ('Compare PDFs', 'Chat with LLM', 'Chat with PDF' ))

################################################################################################

    if option == 'Compare PDFs':
        st.title("📂 Document comparison chatbot")
        uploaded_file1 = st.sidebar.file_uploader("Choose the first PDF file", type="pdf", key="file1")
    
        if uploaded_file1 is not None:
            st.sidebar.success('First PDF uploaded successfully! 🎉')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file1:
                tmp_file1.write(uploaded_file1.read())
                pdf_path1 = tmp_file1.name
    
        uploaded_file2 = st.sidebar.file_uploader("Choose the second PDF file", type="pdf", key="file2")
    
        if uploaded_file2 is not None:
            st.sidebar.success('Second PDF uploaded successfully! 🎉')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file2:
                tmp_file2.write(uploaded_file2.read())
                pdf_path2 = tmp_file2.name
    
        # Main Chat Area
        if (uploaded_file1 is not None) and (uploaded_file2 is not None):
            
            if "history" not in st.session_state:
                st.session_state.history = []
    
            for message in st.session_state.history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    
            query = st.chat_input("Type something... 💬")
            if query:
                st.chat_message("user").markdown(query)
                st.session_state.history.append({"role": "user", "content": query})
    
                with st.spinner('Retrieving information from the first PDF (One time process)⏳'):
                    retriever1, retriever2 = get_retrievers(pdf_path1)
                with st.spinner('Retrieving information from the second PDF (One time process)⏳'):
                    retriever3, retriever4 = get_retrievers(pdf_path2)
                with st.spinner('Comparing answers... ⏳'):
                    response = compare_answers(query, retriever1, retriever2, retriever3, retriever4)
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.history.append({"role": "assistant", "content": response})
        else:
            st.write("Please upload two PDF files to start the comparison chat")

################################################################################################

    elif option == 'Chat with LLM':
        st.title("💬 Chat with LLM")

        llm = ChatGroq(groq_api_key = st.secrets["API_KEY"]["GROQ"], model='llama3-70b-8192', temperature=0.05)
        
        if "history" not in st.session_state:
            st.session_state.history = []
        
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        query = st.chat_input("Type something... ", key="user_input")
            
        if query:
            st.chat_message("user").markdown(query)
            st.session_state.history.append({"role": "user", "content": query})
            st.session_state.input_disabled = True
    
            context = ""
            for pair in st.session_state.history[-10:]:
                context += f"{pair['role']}: {pair['content']}\n"
    
            context += f"user: {query}\n"
    
            # Invoke the LLM with the context
            response = llm.invoke(context).content
    
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.history.append({"role": "assistant", "content": response})
            st.session_state.input_disabled = False 
            
################################################################################################
    
    elif option == 'Chat with PDF':
        st.title("📄 Chat with PDF")

        uploaded_file1 = st.sidebar.file_uploader("Choose the first PDF file", type="pdf", key="file1")
    
        if uploaded_file1 is not None:
            st.sidebar.success('First PDF uploaded successfully! 🎉')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file1:
                tmp_file1.write(uploaded_file1.read())
                pdf_path1 = tmp_file1.name
    
        # Main Chat Area
        if uploaded_file1 is not None:
            
            if "history" not in st.session_state:
                st.session_state.history = []
    
            for message in st.session_state.history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    
            query = st.chat_input("Ask something... ")
            if query:
                st.chat_message("user").markdown(query)
                st.session_state.history.append({"role": "user", "content": query})
    
                with st.spinner('Retrieving information from the PDF...⏳'):
                    retriever1, retriever2 = get_retrievers(pdf_path1)
                    response = get_answer(query, retriever1, retriever2)
                
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.history.append({"role": "assistant", "content": response})
        else:
            st.write("Please upload a PDF file to start the chat")

if __name__ == "__main__":
    main()
