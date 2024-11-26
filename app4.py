import os
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import heapq
from groq import Groq



# Embedding cache management
def generate_or_load_embeddings(data, model, cache_file='embeddings_cache.npz'):
    """
    Generate or load embeddings for the dataset.

    Args:
        data (list): List of dictionaries with 'topic' and 'content'
        model (SentenceTransformer): Embedding model
        cache_file (str): Path to cache file

    Returns:
        tuple: (topic_embeddings, content_embeddings)
    """
    # Check if cache exists
    if os.path.exists(cache_file):
        try:
            # Load existing cache
            cached_data = np.load(cache_file, allow_pickle=True)
            cached_topic_embeddings = cached_data['topic_embeddings']
            cached_content_embeddings = cached_data['content_embeddings']

            # Verify cache matches current data length
            if len(cached_topic_embeddings) == len(data):
                print("Loading embeddings from cache...")
                return cached_topic_embeddings, cached_content_embeddings
        except Exception as e:
            print(f"Error loading cache: {e}")

    # Generate embeddings if no valid cache exists
    print("Generating new embeddings...")
    topic_embeddings = model.encode([item['topic'] for item in data], convert_to_tensor=False)
    content_embeddings = model.encode([item['content'] for item in data], convert_to_tensor=False)

    # Save to cache
    try:
        np.savez_compressed(cache_file,
                            topic_embeddings=topic_embeddings,
                            content_embeddings=content_embeddings)
    except Exception as e:
        print(f"Error saving embeddings cache: {e}")

    return topic_embeddings, content_embeddings


# Retrieval pipeline
def retrieve(query, data, topic_embeddings, content_embeddings, model, top_n=5):
    """
    Retrieve most relevant contexts based on query.
    """
    # Ensure query embedding is on the same device and converted correctly
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Explicitly move query_embedding to CPU if needed
    query_embedding = query_embedding.cpu()

    # Calculate similarities for topics and content
    similarities = []
    for idx, (topic_emb, content_emb) in enumerate(zip(topic_embeddings, content_embeddings)):
        # Convert to tensor and ensure CPU placement
        topic_tensor = torch.tensor(topic_emb).cpu()
        content_tensor = torch.tensor(content_emb).cpu()

        # Calculate similarities
        topic_similarity = util.cos_sim(query_embedding, topic_tensor).item()
        content_similarity = util.cos_sim(query_embedding, content_tensor).item()

        # Combined similarity with higher weight on content
        combined_similarity = (0.3 * topic_similarity) + (0.7 * content_similarity)

        similarities.append((combined_similarity, idx))

    # Select top N results
    top_results = heapq.nlargest(top_n, similarities, key=lambda x: x[0])

    # Prepare results
    results = [
        (similarity, data[idx])
        for similarity, idx in top_results
    ]

    return results, query_embedding


# Function to calculate context relevance
def is_context_relevant(query, result, query_embedding, topic_embedding, content_embedding, threshold=0.4):
    """
    Determine if the retrieved context is relevant to the query.
    """
    # Ensure CPU placement for tensors
    query_embedding = query_embedding.cpu()
    topic_tensor = torch.tensor(topic_embedding).cpu()
    content_tensor = torch.tensor(content_embedding).cpu()

    # Calculate similarities
    topic_similarity = util.cos_sim(query_embedding, topic_tensor).item()
    content_similarity = util.cos_sim(query_embedding, content_tensor).item()

    # Combined similarity with higher weight on content
    combined_similarity = (0.3 * topic_similarity) + (0.7 * content_similarity)

    return combined_similarity > threshold


# Prepare conversation history with a token limit
def prepare_conversation_history(messages, max_tokens=4000):
    """
    Prepare conversation history while respecting token limit.

    Args:
        messages (list): List of message dictionaries
        max_tokens (int): Maximum token limit for history

    Returns:
        list: Trimmed conversation history
    """
    history = []
    current_tokens = 0

    # Iterate in reverse to keep the most recent messages
    for msg in reversed(messages):
        # Estimate token count (rough approximation)
        msg_tokens = len(msg['content'].split())

        if current_tokens + msg_tokens > max_tokens:
            break

        # Prepend to maintain chronological order
        history.append({"role": msg['role'], "content": msg['content']})
        current_tokens += msg_tokens

    return history


def initialize_session_state():
    """
    Initialize all required session state variables.
    """
    # Initialize messages if not exists
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Initialize Groq model if not exists
    if 'groq_model' not in st.session_state:
        st.session_state.groq_model = "llama3-70b-8192"


def main():
    # Initialize session state variables first
    initialize_session_state()

    # Sentence Transformer model and embeddings
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Load data from JSON file
    with open("data.json", "r") as f:
        data = json.load(f)

    # Generate/Load embeddings
    topic_embeddings, content_embeddings = generate_or_load_embeddings(data, model)

    # Initialize Groq client
    client = Groq(api_key=os.getenv("groq_key"))

    st.title("Hydraulic Engineering Assistant")

    # Sidebar for options
    st.sidebar.title("Options")

    # Dropdown for model selection
    model_options = [
        "llama3-70b-8192",
        "gemma-7b-it",
        "gemma2-9b-it",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "llama-3.1-70b-versatile",
    ]
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    st.session_state.groq_model = selected_model

    st.sidebar.write(f"Current Model: **{st.session_state.groq_model}**")

    # Clear chat history button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.sidebar.success("Chat history cleared.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User prompt handling
    if prompt := st.chat_input("Ask a Hydraulic Engineering question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant topics and content
        retrieved_results, query_embedding = retrieve(
            prompt, data, topic_embeddings, content_embeddings, model
        )

        # Filter and prepare relevant contexts
        relevant_contexts = []
        for similarity, result in retrieved_results:
            if is_context_relevant(
                    prompt, result,
                    query_embedding,
                    topic_embeddings[data.index(result)],
                    content_embeddings[data.index(result)]
            ):
                relevant_contexts.append((similarity, result))

            # Stop after two relevant contexts
            if len(relevant_contexts) == 2:
                break

        # Prepare system prompt
        if relevant_contexts:
            enriched_context = "Relevant sections taught by the professor:\n\n"
            for _, result in relevant_contexts:
                enriched_context += (
                    f"**Topic:** {result['topic']}\n"
                    f"**Content:** {result['content']}\n\n"
                )
            enriched_context += "Now, based on these contexts, here is my response:"
            use_context = True
        else:
            enriched_context = "No relevant sections were found in the course material."
            use_context = False

        # Prepare system prompt with conditional context
        system_prompt = (
            f"You are an AI Teaching Assistant for the course Hydraulic Engineering taught by Prof. Saud Afzal at IIT Kharagpur. "
            f"Your answers should be clear, concise, and focused on simplifying complex concepts for students.\n\n"
            f"{enriched_context if use_context else 'Provide a general response to the query without assuming any specific context.'}"
        )

        # Generate response from Groq
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                # Prepare conversation history
                conversation_history = prepare_conversation_history(
                    st.session_state.messages[:-1]  # Exclude the current message
                )

                # Construct messages with history, system prompt, and current query
                messages = [
                               {"role": "system", "content": system_prompt}
                           ] + conversation_history + [
                               {"role": "user", "content": prompt}
                           ]

                # Fetch response from Groq with streaming
                for chunk in client.chat.completions.create(
                        model=st.session_state.groq_model,
                        messages=messages,
                        stream=True,
                ):
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")  # Typing effect

                # Finalize response
                response_placeholder.markdown(full_response)
            except Exception as e:
                response_placeholder.markdown(f"Error: {str(e)}")

            # Add assistant response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})


# Run the Streamlit app
if __name__ == "__main__":
    main()