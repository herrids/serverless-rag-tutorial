# NurseLM Chatbot

NurseLM Chatbot is a sophisticated application that integrates OpenAI's RAG for querying documents, LangChain for conversational AI, and FAISS as a vector database. It features a user-friendly interface powered by Gradio and is encapsulated in Docker for ease of deployment and scalability.

## Features

- **OpenAI's RAG Integration**: Leverages Retrieval-Augmented Generation for effective document querying.
- **LangChain for Conversational AI**: Enhances the chatbot's conversational capabilities.
- **FAISS Vector Database**: Utilizes the FAISS library for efficient similarity search and clustering of dense vectors.
- **Gradio Interface**: Offers an intuitive user interface for interacting with the chatbot.
- **Docker Encapsulation**: Ensures consistent environments and simplifies deployment.

## File Structure

- `prompts.json`: Add system prompt instructions here.
- `data/`: This directory will hold all uploaded PDFs.
- `db/`: Contains FAISS embeddings.
- `src/`: Holds the application source code

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
   docker build -t nurselm .
   ```

2. **Start the Docker Container**:
   ```bash
   docker run -d \
       --env-file .env \
       --name illmy \
       -p 7860:7860 \
       --add-host host.docker.internal:host-gateway \
       -v ./prompts.json:/usr/src/app/prompts.json \
       -v ./db:/usr/src/app/db \
       -v ./data:/usr/src/app/data \
       illmy
   ```

   This command will start the Nurselm Chatbot service on port 7860.

## Usage

Once the Docker container is up and running, access the Gradio interface by navigating to `http://localhost:7860` in your web browser. From here, you can interact with the Nurselm Chatbot and explore its capabilities.