# RAG
Chat with your pdfs using Retrieval Augmented Generation.
![image](https://github.com/Alcatraz2141/RAG/assets/83905457/9053cff5-7bba-4c1e-bf8d-49fefb80e3cb)
## Overview
This Retrieval Augmented Generation (RAG) is a project aimed at optimizing the outputs of LLMS by incorporating a retrieval-based approach. Unlike traditional generative models that generate responses from scratch, RAG systems first retrieve relevant information from a large corpus and then use this retrieved information to guide the generation process. This approach enables the generation of more coherent, contextually relevant, and factually accurate responses.

![Screenshot (41)](https://github.com/Alcatraz2141/RAG/assets/83905457/5cea1753-ba14-4cfe-90f9-7698390fb80d)
## Getting Started
To run the  ChatBot project locally, follow these steps:

1. ##Clone the Repository##: git clone {repo url}

2. ##Install Dependencies##: Navigate to the project directory and install the required dependencies: pip install -r requirements.txt

3. ##Add Google API Key##: Obtain a OPENAI API key and add it to the .env file in the project directory: OPENAI_API_KEY=your_openai_api_key

4. ##Run the Project##: Launch the chatbot application using Streamlit: streamlit run rag.py

5. ##Interact with the ChatBot##: Open your web browser and navigate to the provided URL to interact with  ChatBot.

## Project Structure
The project structure is organized as follows:

-rag.py: Main application file containing the Streamlit user interface and chatbot functionality.

-requirements.txt: List of Python dependencies required to run the project.

- .env: Environment configuration file for storing sensitive information (e.g., API keys).

- README.md: Project documentation file.

