import spacy

# Load the medium-sized English model with word vectors
nlp = spacy.load("en_core_web_md")

# Define the words
word1 = nlp("bought")
word2 = nlp("acquired")

# Compute cosine similarity
cosine_similarity = word1.similarity(word2)

print(f"Cosine similarity between '{word1}' and '{word2}': {cosine_similarity:.4f}")
