import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns




nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_ngrams(text, n):
    tokens = text.split()
    ngrams = set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return ngrams

def jaccard_similarity(set1, set2):
    return float(len(set1.intersection(set2))) / len(set1.union(set2)) if set1.union(set2) else 0.0

def preprocess(text):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(lemmatizer.lemmatize(w)) for w in tokens]
    return cleaned, tokens, " ".join(tokens)


st.title(" Plagiarism & Text Similarity Detection")

uploaded_file1 = st.file_uploader("Upload Text File 1", type=["txt"])
uploaded_file2 = st.file_uploader("Upload Text File 2", type=["txt"])

if uploaded_file1 and uploaded_file2:
    text1 = uploaded_file1.read().decode("utf-8")
    text2 = uploaded_file2.read().decode("utf-8")

    cleaned_text1, tokens1, text1_for_vector = preprocess(text1)
    cleaned_text2, tokens2, text2_for_vector = preprocess(text2)

  
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1_for_vector, text2_for_vector])
    similarity = cosine_similarity(vectors)[0][1]
    plagiarism_percentage = similarity * 100

    common_tokens = set(tokens1).intersection(set(tokens2))
    bigrams1 = get_ngrams(text1_for_vector, 2)
    bigrams2 = get_ngrams(text2_for_vector, 2)
    trigrams1 = get_ngrams(text1_for_vector, 3)
    trigrams2 = get_ngrams(text2_for_vector, 3)
    bigram_jaccard = jaccard_similarity(bigrams1, bigrams2)
    trigram_jaccard = jaccard_similarity(trigrams1, trigrams2)


    st.subheader(" Plagiarism Detection Report")
    st.write(f"**Similarity Score:** {similarity:.4f}")
    st.write(f"**Plagiarism Percentage:** {plagiarism_percentage:.2f} %")

    st.subheader(" Cleaned Texts & Tokens")
    st.write("**Text 1:**", cleaned_text1)
    st.write("Tokens 1:", tokens1)
    st.write("**Text 2:**", cleaned_text2)
    st.write("Tokens 2:", tokens2)

    st.subheader(" Common Words")
    st.write(common_tokens)

    st.subheader(" N-grams Jaccard Similarity")
    st.write(f"Bigrams Jaccard Similarity: {bigram_jaccard:.2f}")
    st.write(f"Trigrams Jaccard Similarity: {trigram_jaccard:.2f}")
    st.write("Common Bigrams:", bigrams1.intersection(bigrams2))
    st.write("Common Trigrams:", trigrams1.intersection(trigrams2))

 
    st.subheader(" TF-IDF Cosine Similarity Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(cosine_similarity(vectors), annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=["Text1", "Text2"], yticklabels=["Text1", "Text2"], ax=ax)
    st.pyplot(fig)

  
    st.subheader(" Word Cloud of Common Words")
    if common_tokens:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(common_tokens))
        fig_wc, ax_wc = plt.subplots(figsize=(10,5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.write("No common words to generate Word Cloud.")
