import pandas as pd
from .content_base import content_based_recommendation
from .collab_base import collaborative_filtering
import os
file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up one level
path = os.path.join(file_path, 'data', 'clean_data.tsv')
train_data = pd.read_csv(path, sep='\t')


def hybrid_recommendations(train_data, target_user_id, item_name=None, top_n=10):
    content_based_rec = content_based_recommendation(train_data, target_user_id, top_n)
    collaborative_filtering_rec = collaborative_filtering(train_data, target_user_id, top_n)

    # Merge results
    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates(subset=["item_name"])
    hybrid_rec = hybrid_rec.sort_values(by="score", ascending=False).head(top_n)

    return hybrid_rec

# Example usage
target_user_id = 4
item_name = "OPI Nail Lacquer Polish .5oz/15mL - This Gown Needs A Crown NL U11"

hybrid_rec = hybrid_recommendations(train_data, target_user_id, item_name, top_n=10)

print(f"\nTop 10 Hybrid Recommendations for User {target_user_id} and Item '{item_name}':\n")
print(hybrid_rec)