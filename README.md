# Document Comparison Chatbot for Regulatory Compliance

## Introduction

The **Document Comparison Chatbot for Regulatory Compliance** project automates the comparison of regulatory documents across countries. It helps companies in the automotive industry comply with varying regulations when entering new markets. The chatbot accepts two regulatory documents and a user query, then generates answers and provides a comparative analysis, highlighting similarities and differences.

## Project Overview

**Background and Motivation**:
- Each country has its own vehicle regulations that must be followed for legal sales.
- Manual comparison of documents from different countries is difficult.

**Objective**: Develop a chatbot to compare legal documents from different countries based on user queries and generate accurate responses to support regulatory compliance.

**Sub-Objective**: Generate precise and contextually accurate responses to user queries.

## The Why

**Motivations**:
- Reduce the time and cost of manual document comparison.
- Ensure regulatory adherence with accuracy.
- Overcome language barriers.

The **homologation department** faces challenges in efficiently comparing regulatory documents across languages. This project proposes a chatbot for the homologation department, automating document comparison using LLM and language translation.

## Approaches and Challenges

### 1. Retrieval-Augmented Generation (RAG)
- **Approach**: Used RAG to retrieve relevant text segments and generate answers.
- **Results**: Provided accurate responses but struggled with incomplete answers.
- **Challenges**: Difficulty maintaining context due to complex PDF structures.

### 2. Page-wise Cumulative Summarization
- **Approach**: Generated summaries page by page, considering previous page summaries.
- **Results**: Improved context but was time-consuming.
  
### 3. RAG with Parent Document Retriever
- **Approach**: Used smaller chunks for retrieval and larger chunks for answer generation.
- **Results**: Improved accuracy but fragmented context.

### 4. Section-wise Chunking
- **Approach**: Treated each section as an independent chunk.
- **Results**: Preserved section integrity but struggled with large sections.

### 5. Custom Multi-Retriever RAG
- **Approach**: Used two retrievers for handling small and large chunks.
- **Results**: Achieved high accuracy by optimizing retrieval.

### 6. DSPy for Optimized Prompts
- **Approach**: Fine-tuned prompts using DSPy to improve LLM's responses.
- **Results**: Improved response accuracy and relevance.

## Major Challenges
1. **Complex PDF Structure**: PDFs with nested sections made maintaining context challenging.
2. **Optimal Chunk Size**: Finding the right chunk size for various questions was difficult.
3. **Dispersed Information**: Retrieving and compiling information from multiple sections complicated the process.

## Evaluation

Manual evaluation using the following metrics:
- **High Accuracy**: Ensured correct facts in answers.
- **High Recall**: Covered all relevant details.
- **High Precision**: Avoided irrelevant information.

## Results

The chatbot provided mostly accurate results, with scope for improvements in DSPy implementation.

## Application

Access the application [here](https://document-comparison-chatbot.streamlit.app/).

**High Latency Reasons**:
- Complex workflow with 4 vector stores and 7 LLM calls.
  
**Potential Solutions**: Switching to cloud services like Claude from Bedrock can reduce response times.

## Future Improvements

- Enhance multi-language support and build a robust knowledge database.
- Improve computational efficiency using advanced techniques and cloud services.
- Further refine DSPy implementation for better performance.

## Conclusion

The Document Comparison Chatbot project has made strides in automating the compliance verification process. Despite time constraints, notable improvements were achieved, with further potential for efficiency and capability expansion in the future.

## Libraries and Tools

- **Libraries**: `langchain`, `pypdf`, `langchain_groq`, `sentence_transformers`, `pdfplumber`, `faiss-cpu`, `streamlit`
- **Tools**: FAISS, RecursiveCharacterTextSplitter, HuggingFace BGE Embeddings, PromptTemplate, DSPy, RetrievalQA
