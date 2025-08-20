import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from functools import lru_cache


class DataLoader:
    def __init__(self, file_name="data.tsv"):
        self.file_name = file_name
        self.data = None

    def load(self):
        file_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(file_path, self.file_name)
        self.data = pd.read_csv(path, sep="\t")
        return self.data


class Preprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data  # ⚡ removed .copy()

    def clean(self):
        cols = [
            "Uniq Id", "Product Id", "Product Rating", "Product Reviews Count",
            "Product Category", "Product Brand", "Product Name",
            "Product Image Url", "Product Description", "Product Tags"
        ]
        self.data = self.data[cols]

        # Fill missing values
        self.data[["Product Rating", "Product Reviews Count"]] = (
            self.data[["Product Rating", "Product Reviews Count"]].fillna(0)
        )
        self.data[["Product Category", "Product Brand", "Product Description"]] = (
            self.data[["Product Category", "Product Brand", "Product Description"]].fillna("")
        )

        # Rename columns
        mapping = {
            "Uniq Id": "ID",
            "Product Id": "ProdID",
            "Product Rating": "Rating",
            "Product Reviews Count": "Reviews",
            "Product Category": "Category",
            "Product Brand": "Brand",
            "Product Name": "Name",
            "Product Image Url": "ImageURL",
            "Product Description": "Description",
            "Product Tags": "Tags",
        }
        self.data.rename(columns=mapping, inplace=True)

        # Convert to numeric IDs
        self.data["ID"] = self.data["ID"].astype(str).str.extract(r"(\d+)").astype(float)
        self.data["ProdID"] = self.data["ProdID"].astype(str).str.extract(r"(\d+)").astype(float)

        return self.data


class EDA:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def dataset_stats(self):
        num_users = self.data["ID"].nunique()
        num_items = self.data["ProdID"].nunique()
        num_ratings = self.data["Rating"].nunique()
        print(f"Users: {num_users}, Items: {num_items}, Ratings: {num_ratings}")

    def plot_heatmap(self):
        heatmap_data = self.data.pivot_table("ID", "Rating")
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt="g", cmap="coolwarm", cbar=True)
        plt.title("Heatmap of User Ratings")
        plt.show()

    def plot_distributions(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        self.data["ID"].value_counts().hist(bins=10, edgecolor="k")
        plt.title("Interactions per User")

        plt.subplot(1, 2, 2)
        self.data["ProdID"].value_counts().hist(bins=10, edgecolor="k", color="orange")
        plt.title("Interactions per Item")
        plt.tight_layout()
        plt.show()

    def plot_popular_items(self):
        popular = self.data["ProdID"].value_counts().head(5)
        popular.plot(kind="bar", color="red")
        plt.title("Most Popular Items")
        plt.show()

    def plot_rating_distribution(self):
        self.data["Rating"].value_counts().plot(kind="bar", color="green")
        plt.title("Rating Distribution")
        plt.show()


class Tagger:
    def __init__(self, data: pd.DataFrame):
        self.data = data  # ⚡ removed .copy()
        # ⚡ Disable heavy components
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "tagger"])
        self.stopwords = set(STOP_WORDS)

    @lru_cache(maxsize=None)  # ⚡ caching duplicate values
    def process_text(self, text: str):
        doc = self.nlp(str(text).lower())
        return ",".join([tok.text for tok in doc if tok.is_alpha and tok.text not in self.stopwords])

    def clean_and_extract_tags_batch(self, texts):
        results = []
        # ⚡ multiprocessing + batch for speed
        for doc in self.nlp.pipe((str(t).lower() for t in texts), 
                                 batch_size=500, n_process=2):
            tags = [tok.text for tok in doc if tok.is_alpha and tok.text not in self.stopwords]
            results.append(",".join(tags))
        return results

    def create_tags(self):
        for col in ["Category", "Brand", "Description"]:
            # ⚡ caching + batch processing
            unique_vals = self.data[col].unique()
            cache_map = {val: self.process_text(val) for val in unique_vals}
            self.data[col] = self.data[col].map(cache_map)

        self.data["Tags"] = self.data[["Category", "Brand", "Description"]].agg(",".join, axis=1)
        return self.data


class IOHandler:
    @staticmethod
    def save_clean_data(data: pd.DataFrame, file_name="clean_data.tsv"):
        file_path = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(file_path, file_name)
        data.to_csv(output_path, sep="\t", index=False)
        print(f"✅ Cleaned data saved to {output_path}")
        return output_path


class RecommenderPipeline:
    """Master pipeline that runs load → preprocess → EDA → tag → save"""

    def __init__(self, input_file="data.tsv", output_file="clean_data.tsv"):
        self.input_file = input_file
        self.output_file = output_file
        self.data = None

    def run(self, run_eda=False, run_plots=False):
        # Load
        loader = DataLoader(self.input_file)
        self.data = loader.load()
        print("📥 Data loaded.")

        # Preprocess
        self.data = Preprocessor(self.data).clean()
        print("🧹 Data preprocessed.")

        # EDA
        if run_eda:
            eda = EDA(self.data)
            eda.dataset_stats()
            if run_plots:
                eda.plot_distributions()
                eda.plot_popular_items()
                eda.plot_rating_distribution()

        # Tagging
        self.data = Tagger(self.data).create_tags()
        print("🏷️ Tags created.")

        # Save
        IOHandler.save_clean_data(self.data, self.output_file)
        print("✅ Pipeline completed.")

        return self.data
 