import pandas as pd
import os
# from data.data_process import load_clean_data
# from recommend_utils import RecommenderPipeline

# from recommend_utils import RecommenderPipeline

# pipeline = RecommenderPipeline("data.tsv", "clean_data.tsv")
# df = pipeline.run(run_eda=True, run_plots=True)

file_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(file_path,'data','clean_data.tsv')
train_data = pd.read_csv(path, sep='\t')
# print(train_data.columns)

#Displaying the top rated items
top_rated_items = train_data.groupby(['Name','Reviews','Brand','ImageURL'])['Rating'].mean().reset_index().sort_values(by='Rating', ascending=False)
rate_base_recommendations = top_rated_items.head(10)
rate_base_recommendations['Rating'] = rate_base_recommendations['Rating'].astype(int)
rate_base_recommendations['Reviews'] = rate_base_recommendations['Reviews'].astype(int)
print("Top 10 Rated Items:")
print(rate_base_recommendations)
print(train_data.columns)