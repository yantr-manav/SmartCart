import pandas as pd
import os 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

file_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(file_path, 'data', 'clean_data.tsv')
train_data = pd.read_csv(path, sep='\t')

def collaborative_filtering(train_data, target_user_id, top_n=10):
    
    user_item_matrix= train_data.pivot_table(index='ID', columns='ProdID',values='Rating',aggfunc='mean').fillna(0).astype(int)

    user_similarity = cosine_similarity(user_item_matrix)
    
# Find the index of the target user in the matrix
    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    
    user_similarities = user_similarity[target_user_index]

    similar_user_indices = user_similarities.argsort()[::-1][1:]
    
    #Generate recommendations based on similar users
    recommend_items = []
    
    for user_index in similar_user_indices:
        
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user==0) & (user_item_matrix.iloc[target_user_index]== 0 )

        recommend_items.extend(user_item_matrix.columns[not_rated_by_target_user][: top_n])
        

    recommended_items_details = train_data[train_data['ProdID'].isin(recommend_items)][['Name', 'Reviews','Brand', 'ImageURL', 'Rating']] 

    return recommended_items_details.head(top_n)


target_user_id = 4
top_n = 5
collaborative_filtering_rec = collaborative_filtering(train_data, target_user_id,top_n)
print(f"Top {top_n} recommendations for User {target_user_id}:")
print(collaborative_filtering_rec)