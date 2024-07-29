import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from concurrent.futures import ProcessPoolExecutor
import langdetect

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Args:
        text (str): Input text.

    Returns:
        str: Detected language code.
    """
    try:
        return langdetect.detect(text)
    except:
        return 'en'  # Default to English if detection fails

def embed_sentences(sentences: List[str], language: str) -> np.ndarray:
    """
    Generate embeddings for a list of sentences using Sentence-BERT.

    Args:
        sentences (List[str]): List of sentences to embed.
        language (str): Language code for the input text.

    Returns:
        np.ndarray: Array of sentence embeddings.
    """
    # Select appropriate model based on language
    if language.startswith('zh'):
        model_name = 'distiluse-base-multilingual-cased-v2'  # Better for chinese
    else:
        model_name = 'all-MiniLM-L6-v2' 
    model = SentenceTransformer(model_name)
    return model.encode(sentences, show_progress_bar=False)

def compute_gap_scores(embeddings: np.ndarray, n: int) -> List[float]:
    """
    Compute gap scores between consecutive sequences of embeddings.

    Args:
        embeddings (np.ndarray): Array of sentence embeddings.
        n (int): Number of sentences to consider in each sequence.

    Returns:
        List[float]: List of gap scores.
    """
    gap_scores = []
    for i in range(n, len(embeddings) - n):
        s_before = embeddings[i-n:i]
        s_after = embeddings[i:i+n]
        sim_score = cosine_similarity(s_before.mean(axis=0).reshape(1, -1), 
                                      s_after.mean(axis=0).reshape(1, -1))[0][0]
        gap_scores.append(1 - sim_score)  # Convert similarity to distance
    #plot_gap_scores(gap_scores)
    return gap_scores

def plot_gap_scores(gap_scores: List[float]):
    """
    Plot the gap scores using Matplotlib.

    Args:
        gap_scores (List[float]): List of gap scores.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(gap_scores)
    plt.title('Cosine Distances Between Sequential Embedding Pairs')
    plt.xlabel('Index')
    plt.ylabel('Cosine Distance')
    plt.grid(True)
    plt.show()


def smooth_scores(scores: List[float], k: int) -> List[float]:
    """
    Apply smoothing to the gap scores using a moving average.

    Args:
        scores (List[float]): List of gap scores.
        k (int): Window size for smoothing.

    Returns:
        List[float]: List of smoothed gap scores.
    """
    smoothed = []
    for i in range(len(scores)):
        start = max(0, i - k // 2)
        end = min(len(scores), i + k // 2 + 1)
        smoothed.append(np.mean(scores[start:end]))
    #plot_gap_scores(smoothed)
    return smoothed

def detect_boundaries(smoothed_scores: List[float], c: float) -> List[int]:
    """
    Detect segment boundaries based on smoothed gap scores.

    Args:
        smoothed_scores (List[float]): List of smoothed gap scores.
        c (float): Threshold parameter for boundary detection.

    Returns:
        List[int]: List of detected boundary indices.
    """
    depth_scores = []
    for i in range(1, len(smoothed_scores) - 1):
        depth = min(smoothed_scores[i] - smoothed_scores[i-1],
                    smoothed_scores[i] - smoothed_scores[i+1])
        depth_scores.append(depth)
    
    mean_depth = np.mean(depth_scores)
    std_depth = np.std(depth_scores)
    threshold = mean_depth + c * std_depth
    
    boundaries = [i + 1 for i, score in enumerate(depth_scores) if score > threshold]
    return boundaries

def cluster_segments(segments: List[str], embeddings: np.ndarray, n_clusters: int) -> List[int]:
    """
    Cluster segments to identify repeated topics.

    Args:
        segments (List[str]): List of text segments.
        embeddings (np.ndarray): Array of segment embeddings.
        n_clusters (int): Number of clusters to form.

    Returns:
        List[int]: Cluster labels for each segment.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(embeddings)

def segment_document(document: str, n: int = 2, k: int = 5, c: float = 1.0) -> List[Tuple[str, int]]:
    """
    Segment a document into coherent chunks using the described method.

    Args:
        document (str): The input document to be segmented.
        n (int): Number of sentences to consider in each sequence for gap score computation.
        k (int): Window size for smoothing.
        c (float): Threshold parameter for boundary detection.

    Returns:
        List[Tuple[str, int]]: List of segmented chunks from the document and their cluster labels.
    """
    # Detect language
    # language = detect_language(document)

    # Tokenize sentences using NLTK
    # sentences = sent_tokenize(document, language=language)
    sentences = sent_tokenize(document)
    
    # Generate embeddings
    embeddings = embed_sentences(sentences, language='en')
    
    # Compute gap scores
    gap_scores = compute_gap_scores(embeddings, n)
    
    # Smooth scores
    smoothed_scores = smooth_scores(gap_scores, k)
    
    # Detect boundaries
    boundaries = detect_boundaries(smoothed_scores, c)
    
    # Create chunks based on detected boundaries
    chunks = []
    start = 0
    for boundary in boundaries:
        chunks.append(' '.join(sentences[start:boundary+n]))
        start = boundary + n
    chunks.append(' '.join(sentences[start:]))
    
    # Cluster segments
    n_clusters = min(len(chunks), 5)  # Adjust the number of clusters as needed
    chunk_embeddings = embed_sentences(chunks, language='en')
    cluster_labels = cluster_segments(chunks, chunk_embeddings, n_clusters)
    
    return list(zip(chunks, cluster_labels)), smoothed_scores, boundaries

def process_chunk(args):
    """Helper function for parallel processing."""
    chunk, language = args
    return embed_sentences([chunk], language)[0]

def parallel_embed_sentences(sentences: List[str], language: str) -> np.ndarray:
    """
    Generate embeddings for a list of sentences using parallel processing.

    Args:
        sentences (List[str]): List of sentences to embed.
        language (str): Language code for the input text.

    Returns:
        np.ndarray: Array of sentence embeddings.
    """
    with ProcessPoolExecutor() as executor:
        embeddings = list(executor.map(process_chunk, [(sent, language) for sent in sentences]))
    return np.array(embeddings)

def visualize_segmentation(smoothed_scores: List[float], boundaries: List[int]):
    """
    Visualize the segmentation with smoothed gap scores and segmentation markers.

    Args:
        smoothed_scores (List[float]): List of smoothed gap scores.
        boundaries (List[int]): List of detected boundary indices.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed_scores, label='Smoothed gap score')
    plt.scatter([b for b in boundaries if b < len(smoothed_scores)], 
                [smoothed_scores[b] for b in boundaries if b < len(smoothed_scores)],
                color='red', label='Segmentation markers')
    plt.xlabel('Smoothed gap score indices')
    plt.ylabel('Smoothed gap score')
    plt.title('Document Segmentation Visualization')
    plt.legend()
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation_bars(smoothed_scores: List[float], boundaries: List[int]):
    """
    Visualize the segmentation with smoothed gap scores, segmentation markers, and colored chunks.

    Args:
        smoothed_scores (List[float]): List of smoothed gap scores.
        boundaries (List[int]): List of detected boundary indices.
    """
    plt.figure(figsize=(15, 8))

    # Plot the smoothed scores
    plt.plot(smoothed_scores)

    # Set y-axis limit
    y_upper_bound = max(smoothed_scores) * 1.1
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(smoothed_scores))

    # Calculate threshold (using mean + standard deviation instead of percentile)
    threshold = np.mean(smoothed_scores) + np.std(smoothed_scores)
    plt.axhline(y=threshold, color='r', linestyle='-')

    # Count chunks
    num_chunks = len(boundaries) + 1
    plt.text(x=(len(smoothed_scores)*0.01), y=y_upper_bound/50, s=f"{num_chunks} Chunks")

    # Color and label chunks
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    start_index = 0
    for i, end_index in enumerate(boundaries + [len(smoothed_scores)]):
        plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
        plt.text(x=np.average([start_index, end_index]),
                 y=threshold + (y_upper_bound) / 20,
                 s=f"Chunk #{i+1}", horizontalalignment='center',
                 rotation='vertical')
        start_index = end_index

    plt.title("Document Chunks Based On Embedding Breakpoints")
    plt.xlabel("Index of sentences in document (Sentence Position)")
    plt.ylabel("Smoothed gap score between sequential sentences")
    plt.tight_layout()
    plt.show()

# Modify the segment_document function to return smoothed_scores and boundaries
def segment_document(document: str, n: int = 2, k: int = 5, c: float = 1.0) -> Tuple[List[str], List[float], List[int]]:
    """
    Segment a document into coherent chunks using the described method.

    Args:
        document (str): The input document to be segmented.
        n (int): Number of sentences to consider in each sequence for gap score computation.
        k (int): Window size for smoothing.
        c (float): Threshold parameter for boundary detection.

    Returns:
        List[Tuple[str, int]]: List of segmented chunks from the document and their cluster labels.
    """
    # Detect language
    # language = detect_language(document)

    # Tokenize sentences using NLTK
    # sentences = sent_tokenize(document, language=language)
    sentences = sent_tokenize(document)
    
    # Generate embeddings
    embeddings = embed_sentences(sentences, language='en')
    
    # Compute gap scores
    gap_scores = compute_gap_scores(embeddings, n)
    
    # Smooth scores
    smoothed_scores = smooth_scores(gap_scores, k)
    
    # Detect boundaries
    boundaries = detect_boundaries(smoothed_scores, c)
    
    
    # Create chunks based on detected boundaries
    chunks = []
    start = 0
    for boundary in boundaries:
        chunks.append(' '.join(sentences[start:boundary+n]))
        start = boundary + n
    chunks.append(' '.join(sentences[start:]))
    
    return chunks, smoothed_scores, boundaries


    
if __name__ == "__main__":
    with open('essays.txt', 'r') as file:
        document = file.read()
    chunks, smoothed_scores, boundaries = segment_document(document, n=1, k=1, c=1.0)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print()
    
    visualize_segmentation_bars(smoothed_scores, boundaries)