import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download NLTK's sentence tokenizer if not already downloaded
nltk.download('punkt')

def main():
    st.title('ClauseScribe')
    st.subheader('**Describe**')
    
    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("File uploaded successfully!")
        
        # Text input
        input_text = st.text_area("Input Text Here", height=150)
        
        # Action selection
        selected_action = st.radio("Choose an action:", ('Summary', 'Classification', 'Discrepancy'))
        
        # Perform action based on the selected button
        if st.button('Perform Action'):
            if selected_action == 'Summary':
                summary_result = generate_summary(input_text, data)
                st.write("**Summary:**")
                st.write(summary_result)
            elif selected_action == 'Classification':
                classification_result = perform_classification(input_text, data)
                st.write("**Classification:**")
                st.write(classification_result)
            elif selected_action == 'Discrepancy':
                discrepancy_result = generate_discrepancy(input_text, data)
                st.write("**Discrepancy:**")
                st.write(discrepancy_result)

def generate_summary(legal_text, data):
    sentences = nltk.sent_tokenize(legal_text)
    # Join the first few sentences as summary (adjust this logic as needed)
    summary = ''.join(sentences[:1])  # Extracting the first 3 sentences as a summary
    return summary

def preprocess_text(text):
    # Handle NaN and non-string values, convert to string
    if pd.isnull(text) or not isinstance(text, str):
        return str(text)
    return text

def perform_classification(text, data):
    # Preprocess text
    preprocessed_texts = data['clause_text'].apply(preprocess_text)

    # Initialize and train CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(preprocessed_texts)
    y = data['clause_type']  # Assuming 'label' is the column containing class labels

    # Initialize and train the classifier (Multinomial Naive Bayes)
    classifier = MultinomialNB()
    classifier.fit(X, y)

    # Predict the labels for input text
    input_text_vectorized = vectorizer.transform([text])
    predicted_label = classifier.predict(input_text_vectorized)

    return f'Predicted Label: {predicted_label[0]}'

def generate_discrepancy(text, data):
    # Your discrepancy logic here...
    # Drop rows with missing clause texts
    data = data.dropna(subset=['clause_text'])

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform clause texts
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['clause_text'])

    # Transform the input text
    input_vector = tfidf_vectorizer.transform([text])

    # Calculate cosine similarity with input text
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()

    # Find the most similar clause and its index
    most_similar_index = cosine_similarities.argmax()
    most_similar_clause = data['clause_text'].iloc[most_similar_index]
    similarity_score = cosine_similarities[most_similar_index]

    # Calculate text discrepancies
    text_diff = SequenceMatcher(None, text, most_similar_clause)
    discrepancy_ratio = 1 - text_diff.ratio()

    # Display discrepancy results
    result = ""
    result += f"Cosine Similarity Score: {similarity_score}\n"
    if discrepancy_ratio > 0.2:
        result += f"Text Discrepancy: {discrepancy_ratio}\n"
        result += "\nUnsimilar Text:\n"
        result += most_similar_clause
    else:
        result += f"Text Discrepancy: {discrepancy_ratio}"

    return result

if __name__ == "__main__":
    main()
