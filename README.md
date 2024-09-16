# Document Comparison Chatbot for Regulatory Compliance

## Introduction

Background and Motivation:
- Each country has its own rules and regulations related to vehicles. These rules and regulation must be followed for a legal sale of vehicles.
- 

## Technical Approach

We initially started with RAG (Retrieval-Augmented Generation)
-  Reponses were incomplete


Methodology:
- Information Retrieval: Handling queries by retrieving relevant text sections from legal documents.
- Document Chunking: Breaking down documents into optimal chunk sizes for accurate responses.
- Custom Multi-Retriever RAG: Using multiple retrievers for different chunk sizes to balance detailed and crisp responses.

## Challenges and Solutions

Challenges:
- Complex PDF Formatting: Dealing with deeply nested sections and complex layouts.
- Cumulative Summarization: Addressing the challenge of summarizing large sections of text.
- Optimal Chunk Size: Balancing between retrieving small, precise answers and larger, detailed responses.
- Handling Multiple Languages: Future improvement to support multilingual document comparison.

## Evaluation

Performance Metrics:
- Accuracy, Precision, Recall.

Manual Evaluation:
- Evaluating correctness and completeness of information provided by the chatbot.

Challenges in Evaluation:
- Complex structure of legal documents leading to increased processing time.

## Results and Discussion

Case Studies:
- Example queries and responses.

Observations:
- Strengths and limitations of the current system.

Impact on Regulatory Compliance:
- How the chatbot assists in ensuring compliance across different jurisdictions.

## Future Improvements

Scalability:
- Enhancing the chatbot’s ability to handle more documents and queries simultaneously.

Language Support:
- Incorporating support for multiple languages to broaden applicability.

Knowledge Database:
- Developing a centralized database of regulations for quicker retrieval and response generation.

## Conclusion

Summary of Achievements:
- Successfully automated the document comparison process with accurate, context-aware responses.

Implications for the Industry:
- Potential for broader application in regulatory compliance and legal tech.

## References

Academic Papers:
- List of relevant papers and articles consulted during the project.

Tools and Libraries:
- Description of tools, models, and libraries used.

Team Contributions:
- Detailed description of each team member’s role and contributions.

Additional Documentation:
- Screenshots, code snippets, and additional diagrams explaining the implementation.

