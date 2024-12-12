## Overview

**CourseMate** is a Retrieval-Augmented Generation (RAG) application designed to automate the creation of a query-based assistant for course materials. It retrieves transcripts from YouTube playlists, processes them into a searchable database, and uses semantic similarity to answer user queries contextually.

This project leverages:
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/) and [Pytube](https://pytube.io) for transcript fetching.
- [SentenceTransformers](https://www.sbert.net) for embedding generation.
- [Groq API](https://groq.com) for AI-driven response generation.
- [Streamlit](https://streamlit.io) for a user-friendly interface.

---

## Workflow

### 1. Fetching Transcripts from YouTube

The bot begins by extracting transcripts from a given YouTube playlist. The provided Python script downloads the transcripts for each video in the playlist and saves them in a structured JSON file.

### 2. Data Processing and Database Creation

The extracted transcripts are parsed into a structured format with **topics** and **content**, suitable for embedding generation and semantic retrieval.

### 3. Generating Embeddings

CourseMate uses the `SentenceTransformers` library to create embeddings for both topics and content. These embeddings are cached for efficient retrieval.

- **Function:** `generate_or_load_embeddings`
- **Purpose:** Avoids regenerating embeddings for unchanged data.

### 4. Retrieval and Ranking

When a query is input:

1. The query is embedded using the same model.
2. Similarity is computed between the query embedding and topic/content embeddings.
3. Results are ranked based on a weighted combination of topic and content similarity.

### 5. AI-Powered Response Generation

The bot uses the **Groq API** to generate a context-aware response. It combines:
- Retrieved context from the course material.
- User query.
- System prompt designed to provide clear and concise answers.

### 6. Interactive Chat Interface

The Streamlit-based interface supports:
- Query input and real-time responses.
- Model selection for flexibility.
- Chat history management.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python libraries:
  - `youtube-transcript-api`
  - `pytube`
  - `sentence-transformers`
  - `numpy`
  - `torch`
  - `streamlit`
  - `groq`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/mrbane10/CourseMate.git
   cd course-query-bot

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   
3. Add your Groq API key as an environment variable:
   ```bash
   export groq_key=your_api_key_here
