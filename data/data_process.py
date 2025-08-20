import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix

# File path to the dataset and reading the data
def load_data():
    file_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(file_path,'data.tsv')
    train_data = pd.read_csv(path, sep='\t')
    return train_data

train_data = load_data()
# Selecting relevant columns for the analysis
train_data = train_data[['Uniq Id','Product Id','Product Rating','Product Reviews Count', 'Product Category','Product Brand', 'Product Name','Product Image Url', 'Product Description', 'Product Tags' ]]
# print(train_data)

# Replacing null values in 'Product Rating' and 'Product Reviews Count' with 0 and empty strings in 'Product Category', 'Product Brand', and 'Product Description' 
# ...existing code...
train_data[['Product Rating', 'Product Reviews Count']] = train_data[['Product Rating', 'Product Reviews Count']].fillna(0)
train_data[['Product Category', 'Product Brand', 'Product Description']] = train_data[['Product Category', 'Product Brand', 'Product Description']].fillna('')
# print(train_data.isnull().sum())
# print(train_data.duplicated().sum())

# Defining the mapping for product categories
category_mapping = {
    'Uniq Id': 'ID',
    'Product Id': 'ProdID',
    'Product Rating': 'Rating',
    'Product Reviews Count': 'Reviews',
    'Product Category': 'Category',
    'Product Brand': 'Brand',
    'Product Name': 'Name',
    'Product Image Url': 'ImageURL',
    'Product Description': 'Description',
    'Product Tags': 'Tags'
}
# Renaming the columns in the DataFrame
train_data.rename(columns=category_mapping, inplace=True)
# print(train_data.columns)

#Converting 'ID' and 'Product ID' to numeric type: float
train_data['ID'] = train_data['ID'].str.extract(r'(\d+)').astype(float)
train_data['ProdID'] = train_data['ProdID'].str.extract(r'(\d+)').astype(float)

# print(train_data[['ID','ProdID']])

# Exploratory data analysis
num_users = train_data["ID"].nunique()
num_items = train_data["ProdID"].nunique()
num_ratings = train_data["Rating"].nunique()
print(f"Number of unique users: {num_users}")
print(f"Number of unique items: {num_items}")   
print(f"Number of unique ratings: {num_ratings}")

# heatmap of product ratings
heatmap_data = train_data.pivot_table('ID','Rating')

plt.figure(figsize=(8,6))
sns.heatmap(heatmap_data,annot=True,fmt='g',cmap='coolwarm',cbar=True)
plt.title('heatmap of User Ratings')
plt.xlabel('Rating')
plt.ylabel('User ID')
# plt.show()

# Distribution of Interactions

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
train_data['ID'].value_counts().hist(bins=10,edgecolor='k')
plt.xlabel("Interactions per User")
plt.ylabel("Number of Users")
plt.title("Distribution of Interactions per Users")

plt.subplot(1, 2, 2)
train_data['ProdID'].value_counts().hist(bins=10,edgecolor='k',color='orange')
plt.xlabel("Interactions per Item")
plt.ylabel("Number of Items")
plt.title("Distribution of Interactions per Items")

plt.tight_layout()
# plt.show()

# Most Popular Items

popular_items = train_data['ProdID'].value_counts().head(5)
popular_items.plot(kind='bar',color='red')
plt.title("Most Popular Items")
# plt.show()

# Most Rated Counts

train_data['Rating'].value_counts().plot(kind='bar', color='green')
plt.show()

# Data Cleaning and  Tags Creation

nlp = spacy.load("en_core_web_sm")
columns_to_extract_tags = ['Category','Brand','Description']

def clean_and_extract_tags_batch(texts):
    results = []
    for doc in nlp.pipe([str(t).lower() for t in texts], batch_size=1000):
        tags = [token.text for token in doc if token.text.isalnum() and token.text not in STOP_WORDS]
        results.append(','.join(tags))
    return results

for col in columns_to_extract_tags:
    train_data[col] = clean_and_extract_tags_batch(train_data[col])
    
    
train_data['Tags']= train_data[columns_to_extract_tags].apply(lambda row: ','.join(row),axis=1)
# print(train_data)
# Saving the cleaned data to a new file
def load_clean_data():
    
    file_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(file_path,'clean_data.tsv')
    train_data.to_csv(output_path,sep='\t',index=False)
    print(f"Cleaned data saved to {output_path}")
    return train_data
    