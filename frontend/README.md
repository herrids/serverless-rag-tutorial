# Chatbot Frontend

 Chatbot is a sophisticated application that integrates RAG for querying documents, LangChain for conversational AI, and FAISS as a vector database. It features a user-friendly interface powered by Gradio and is encapsulated in Docker for ease of deployment and scalability.

## Features

- **RAG Integration**: Leverages Retrieval-Augmented Generation for effective document querying.
- **LangChain for Conversational AI**: Enhances the chatbot's conversational capabilities.
- **FAISS Vector Database**: Utilizes the FAISS library for efficient similarity search and clustering of dense vectors.
- **Gradio Interface**: Offers an intuitive user interface for interacting with the chatbot.
- **Docker Encapsulation**: Ensures consistent environments and simplifies deployment.

## Getting Started

### Prerequisites

- Docker installed on your system.
- OpenAI API key.

### Configuration

1. **Set Up Environment Variables**:
   - Add your OpenAI API key to the `env.example` file.
   - Rename the file to `.env`.

### Build and Run with Docker

1. **Build the Docker Container**:
   ```bash
   docker build -t frontend .
   ```

2. **Start the Docker Container**:
   ```bash
   docker run -d \
       --env-file .env \
       --name frontend \
       -p 7860:7860 \
       --add-host host.docker.internal:host-gateway \
       frontend
   ```

   This command will start the Chatbot service on port 7860.

## Usage

Once the Docker container is up and running, access the Gradio interface by navigating to `http://localhost:7860` in your web browser. From here, you can interact with the Chatbot and explore its capabilities.