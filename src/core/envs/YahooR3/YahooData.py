import os
import sys
import pickle
import pandas as pd
import numpy as np

sys.path.extend([".", "./src", "./src/DeepCTR-Torch", "./src/tianshou"])
from src.core.envs.BaseData import BaseData, get_distance_mat

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
ROOTPATH = os.path.join(REPO_ROOT, "data", "YahooR3")
DATAPATH = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")

for path in [PRODATAPATH]:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class YahooData(BaseData):
    def __init__(self):
        super(YahooData, self).__init__()
        self.train_data_path = "ydata-ymusic-rating-study-v1_0-train.txt"
        self.val_data_path = "ydata-ymusic-rating-study-v1_0-test.txt"
        self.questionnaire_path = "ydata-ymusic-rating-study-v1_0-survey-answers.txt"
        
    def get_features(self, is_userinfo=None):
        user_features = [
            "user_id",
            "rate_frequency",
            "rate_hate",
            "rate_dislike",
            "rate_neutral",
            "rate_like",
            "rate_love",
            "preference_sensitive",
            "rating_activeness",
        ]
        item_features = ['item_id']
        reward_features = ["rating"]
        return user_features, item_features, reward_features

    def get_df(self, name="ydata-ymusic-rating-study-v1_0-train.txt"):
        # read interaction
        filename = os.path.join(DATAPATH, name)
        df_data = pd.read_csv(filename, sep="\s+", header=None, names=["user_id", "item_id", "rating"])

        df_data["user_id"] -= 1
        df_data["item_id"] -= 1

        df_user = self.load_user_feat()
        df_item = self.load_item_feat()
        list_feat = None
        df_data = df_data.join(df_user, on="user_id", how="left")
        df_data = df_data.join(df_item, on="item_id", how="left")

        return df_data, df_user, df_item, list_feat

    def get_domination(self):
        return None
    
    def get_item_similarity(self):
        item_similarity_path = os.path.join(PRODATAPATH, "item_similarity.pickle")
        if os.path.isfile(item_similarity_path):
            item_similarity = pickle.load(open(item_similarity_path, 'rb'))
        else:
            mat = YahooData.load_mat()
            mat_distance = YahooData.get_saved_distance_mat(mat, PRODATAPATH)
            item_similarity = 1 / (mat_distance + 1)
            pickle.dump(item_similarity, open(item_similarity_path, 'wb'))
        return item_similarity
      
    def get_item_popularity(self):
        item_popularity_path = os.path.join(PRODATAPATH, "item_popularity.pickle")

        if os.path.isfile(item_popularity_path):
            item_popularity = pickle.load(open(item_popularity_path, 'rb'))
        else:
            df_data, df_user, df_item, list_feat = self.get_df("ydata-ymusic-rating-study-v1_0-train.txt")

            n_users = df_data['user_id'].nunique()
            n_items = df_data['item_id'].nunique()

            df_data_filtered = df_data[df_data["rating"]>=3.]
            
            groupby = df_data_filtered.loc[:, ["user_id", "item_id"]].groupby(by="item_id")
            df_pop = groupby.user_id.apply(list).reset_index()
            df_pop["popularity"] = df_pop['user_id'].apply(lambda x: len(x) / n_users)

            item_pop_df = pd.DataFrame(np.arange(n_items), columns=["item_id"])
            item_pop_df = item_pop_df.merge(df_pop, how="left", on="item_id")
            item_pop_df['popularity'].fillna(0, inplace=True)
            item_popularity = item_pop_df['popularity']
            pickle.dump(item_popularity, open(item_popularity_path, 'wb'))
        
        return item_popularity

    def load_user_feat(self):
        df_user = pd.DataFrame(np.arange(15400), columns=["user_id"])
        df_user.set_index("user_id", inplace=True)
        answers_path = os.path.join(DATAPATH, self.questionnaire_path)
        df_q = pd.read_csv(
            answers_path,
            sep=r"\s+",
            header=None,
            names=[
                "rate_frequency",
                "rate_hate",
                "rate_dislike",
                "rate_neutral",
                "rate_like",
                "rate_love",
                "preference_sensitive",
            ],
            dtype=int,
        )
        # Survey answers are line-aligned with user ids 1..5400 (first 5400 users).
        df_q["user_id"] = np.arange(len(df_q))
        df_q["rating_activeness"] = df_q[
            ["rate_frequency", "rate_hate", "rate_dislike", "rate_neutral", "rate_like", "rate_love"]
        ].mean(axis=1).round().astype(int)
        df_q = df_q.set_index("user_id")

        df_user = df_user.join(df_q, how="left")
        df_user = df_user.fillna(0).astype(int)
        return df_user

    def load_item_feat(self):
        df_item = pd.DataFrame(np.arange(1000), columns=["item_id"])
        df_item.set_index("item_id", inplace=True)
        return df_item


    @staticmethod
    def load_mat():
        # Note: The data file `yahoo_pseudoGT_ratingM.ascii` is sourced from the https://github.com/BetsyHJ/RL4Rec repository.
        filename_GT = os.path.join(DATAPATH, "RL4Rec_data", "yahoo_pseudoGT_ratingM.ascii")
        mat = pd.read_csv(filename_GT, sep="\s+", header=None, dtype=str).to_numpy(dtype=int)
        return mat


if __name__ == "__main__":
    dataset = YahooData()
    df_train, df_user_train, df_item_train, _ = dataset.get_train_data()
    df_val, df_user_val, df_item_val, _ = dataset.get_val_data()
    print("YahooR3: Train #user={}  #item={}  #inter={}".format(df_train['user_id'].nunique(), df_train['item_id'].nunique(), len(df_train)))
    print("YahooR3: Test  #user={}  #item={}  #inter={}".format(df_val['user_id'].nunique(), df_val['item_id'].nunique(), len(df_val)))
    # ===== DEBUG user features =====
    print("\n===== DEBUG df_user_train =====")
    print("shape:", df_user_train.shape)
    print("columns:", df_user_train.columns.tolist())
    print(df_user_train.head())

    print("\nDescribe df_user_train:")
    print(df_user_train.describe(include="all"))

    print("\nSample users:")
    print(df_user_train.sample(5, random_state=0))

    print("\n===== DEBUG df_user_val =====")
    print("shape:", df_user_val.shape)
    print("columns:", df_user_val.columns.tolist())
    print(df_user_val.head())
