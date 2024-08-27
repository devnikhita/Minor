import streamlit as st
import sqlite3
import pandas as pd
import pickle
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Custom CSS for font and size
st.markdown("""
    <style>
    /* Global font settings */
    body {
        font-family: 'Arial', sans-serif;
        font-size: 18px;
    }

    /* Increase font size for the tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 20px;
        font-weight: bold;
    }

    /* Increase font size for titles */
    h1 {
        font-size: 36px !important;
    }

    h2 {
        font-size: 28px !important;
    }

    h3 {
        font-size: 24px !important;
    }

    /* Style for book details */
    .book-details {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Load BERT model and tokenizer
@st.cache_resource
def load_bert_model_and_tokenizer():
    try:
        # Load model and tokenizer from the pickle file
        with open("bert_model.pkl", "rb") as file:
            saved_objects = pickle.load(file)
        
        # Extract model state_dict and tokenizer
        model_state_dict = saved_objects.get('model_state_dict')
        tokenizer = saved_objects.get('tokenizer')

        # Initialize model and load state_dict
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        model.load_state_dict(model_state_dict)
        
        if tokenizer is None:
            # Load tokenizer separately if not included in the pickle
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load BERT model or tokenizer: {e}")
        return None, None

# Database connection
def get_db_connection():
    conn = sqlite3.connect('books.db')
    return conn

# Fetch books from the database
def fetch_books(search_query=""):
    conn = get_db_connection()
    query = "SELECT * FROM books WHERE book_name LIKE ? OR author_name LIKE ? OR genre LIKE ?"
    books = pd.read_sql(query, conn, params=[f'%{search_query}%', f'%{search_query}%', f'%{search_query}%'])
    conn.close()
    return books

# Add a book to a list (wishlist or cart)
def add_to_list(book_id, list_type):
    if list_type not in st.session_state:
        st.session_state[list_type] = []
    if book_id not in st.session_state[list_type]:
        st.session_state[list_type].append(book_id)
        if list_type == 'wishlist':
            st.session_state['new_wishlist_item'] = True  # Flag to trigger recommendation
            recommend_books()

# Display books with heart and cart buttons
def display_books(books):
    for i in range(0, len(books), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(books):
                row = books.iloc[i + j]
                with cols[j]:
                    st.image(row['image_url'], width=150)
                    st.write(f"<div class='book-details'><b>{row['book_name']}</b></div>", unsafe_allow_html=True)
                    st.write(f"<div class='book-details'><i>{row['author_name']}</i></div>", unsafe_allow_html=True)
                    st.write(f"<div class='book-details'><b>Genre:</b> {row['genre']}</div>", unsafe_allow_html=True)
                    st.write(f"<div class='book-details'><b>Published:</b> {row['yop']}</div>", unsafe_allow_html=True)
                    st.write(f"<div class='book-details'><b>Publisher:</b> {row['name_of_publisher']}</div>", unsafe_allow_html=True)

                    # Heart for Wishlist
                    if st.button(f"❤️ Add to Wishlist", key=f"wishlist_{row['id']}"):
                        add_to_list(row['id'], 'wishlist')
                        st.success("Added to Wishlist")

                    # Button for Cart
                    if st.button("🛒 Add to Cart", key=f"cart_{row['id']}"):
                        add_to_list(row['id'], 'cart')
                        st.success("Added to Cart")

                    # View Details Button
                    if st.button("View Details", key=f"view_{row['id']}"):
                        st.write(row['description'])
            st.write("---")

# View books in a list (wishlist or cart)
def view_list(list_type):
    if list_type in st.session_state and st.session_state[list_type]:
        books = fetch_books()
        list_books = books[books['id'].isin(st.session_state[list_type])]
        display_books(list_books)
    else:
        st.write("Your list is empty.")

# Recommend books using BERT model
def recommend_books_bert(book_names):
    model, tokenizer = load_bert_model_and_tokenizer()
    if model is None or tokenizer is None:
        st.write("BERT model or tokenizer not loaded, unable to generate recommendations.")
        return pd.DataFrame()  # Return an empty DataFrame

    # Tokenize and encode the book names
    wishlist_encodings = [tokenizer(name, truncation=True, padding='max_length', max_length=512, return_tensors='pt') for name in book_names]
    wishlist_embeddings = []

    model.eval()
    with torch.no_grad():
        for encoding in wishlist_encodings:
            output = model(**encoding)
            wishlist_embeddings.append(output.logits.squeeze().numpy())

    wishlist_embeddings = np.array(wishlist_embeddings)

    # Fetch dataset embeddings
    books = fetch_books()
    dataset_encodings = [tokenizer(desc, truncation=True, padding='max_length', max_length=512, return_tensors='pt') for desc in books['description']]
    dataset_embeddings = []

    with torch.no_grad():
        for encoding in dataset_encodings:
            output = model(**encoding)
            dataset_embeddings.append(output.logits.squeeze().numpy())

    dataset_embeddings = np.array(dataset_embeddings)
    similarity_scores = cosine_similarity(wishlist_embeddings, dataset_embeddings)

    # Get top recommendations
    top_k = 5
    recommendations = []

    for i, scores in enumerate(similarity_scores):
        top_indices = scores.argsort()[-top_k:][::-1]
        recommended_books = books.iloc[top_indices]
        recommendations.append({
            'wishlist_book': book_names[i],
            'recommended_books': recommended_books['book_name'].values
        })

    return pd.DataFrame(recommendations)

# Generate and display recommendations
def recommend_books():
    if 'wishlist' in st.session_state and st.session_state['wishlist']:
        books = fetch_books()
        wishlist_books = books[books['id'].isin(st.session_state['wishlist'])]

        # Extract book names and generate recommendations
        book_names = wishlist_books['book_name'].tolist()
        st.write(f"Book names in wishlist: {book_names}")  # Debugging print
        recommendations_df = recommend_books_bert(book_names)

        if not recommendations_df.empty:
            st.write("Based on your wishlist, we recommend the following books:")
            for _, rec in recommendations_df.iterrows():
                st.write(f"**Wishlist Book:** {rec['wishlist_book']}")
                st.write("**Top Recommendations:**")
                for book in rec['recommended_books']:
                    st.write(f"- {book}")
                st.write("\n")
        else:
            st.write("No recommendations found.")
    else:
        st.write("Add books to your wishlist to get recommendations.")

# Home page displaying all books
def home_page():
    st.title("ReadNest")
    st.subheader("Where stories find you.")
    books = fetch_books()
    display_books(books)

# Wishlist page displaying books added to wishlist
def wishlist_page():
    st.title("Wishlist")
    view_list('wishlist')
    recommend_books()  # Generate recommendations after viewing wishlist

# Cart page displaying books added to cart
def cart_page():
    st.title("Cart")
    view_list('cart')

# Search page to search for books
def search_page():
    st.title("Search Books")
    search_query = st.text_input("Enter search term...")
    if search_query:
        books = fetch_books(search_query)
        display_books(books)
    else:
        st.write("Please enter a search term.")

# Recommended books page (if triggered manually)
def recommended_page():
    st.title("Recommended Books")
    recommend_books()

# Main app
def main():
    # Sidebar with app name and tagline
    st.sidebar.title("ReadNest")
    st.sidebar.subheader("Where stories find you")

    # Sidebar navigation
    page = st.sidebar.radio("Navigate", ["Home", "Search", "Wishlist", "Cart", "Recommended"])

    # Display the selected page
    if page == "Home":
        home_page()
    elif page == "Search":
        search_page()
    elif page == "Wishlist":
        wishlist_page()
    elif page == "Cart":
        cart_page()
    elif page == "Recommended":
        recommended_page()

if __name__ == "__main__":
    main()