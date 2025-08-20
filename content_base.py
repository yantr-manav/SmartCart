# Content Base Recommendation System
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

file_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(file_path,'data','clean_data.tsv')
train_data = pd.read_csv(path, sep='\t')





# print(cosine_similarities_content)

item_name = 'OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath'



def content_based_recommendation(train_data, item_name, top_n=10):
    # Check if item name exists in the dataset
    if item_name not in train_data['Name'].values:
        raise ValueError(f"Item '{item_name}' not found in the dataset.")
        return pd.DataFrame()
    
    # Create a TF-IDF vectorizer  for item descriptions 
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
  
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content,tfidf_matrix_content)
    
    # Find the index of the item 
    item_index = train_data[train_data['Name']== item_name].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    sorted_similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    
    top_similar_items = sorted_similar_items[1:11]
    recommended_items_indices = [x[0] for x in top_similar_items] 
    recommended_item_details = train_data.iloc[recommended_items_indices][['Name','Reviews','Brand']]
    return recommended_item_details

if __name__ == "__main__":
    item_name = 'Kokie Professional Matte Lipstick, Hot Berry, 0.14 fl oz'
    try:
        recommendations = content_based_recommendation(train_data, item_name)
        print("Content-Based Recommendations:")
        print(recommendations)
    except ValueError as e:
        print(e)