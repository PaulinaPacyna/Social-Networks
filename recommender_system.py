import os
import re
from collections import Counter
from typing import List, Dict

import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

DATA_DIR = "netflix_data"
COMBINED_DATA = "combined_data_{}.txt"

LIMIT_USERS = 500  # I have to limit the possible users or movies, other way the program is computing forever
LIMIT_MOVIES = float("inf")


class Recommender:
    def __init__(self, user_id, n_similar, min_common):
        self.user_id = user_id
        self.n_similar = n_similar
        self.min_common = min_common
        self.my_features = self.__extract_user_data()
        self.similar = []

    def __extract_user_data(self) -> Dict[int, Dict]:
        """Extract movies, ratings and dates for this specific user"""
        data: List[str] = []
        collect = False
        for file_name in [
            os.path.join(DATA_DIR, COMBINED_DATA.format(i + 1)) for i in list(range(4))
        ]:
            with open(file_name) as f:
                for line in f:
                    line = line.strip()
                    if re.match(r"\d+:", line):
                        if collect:  # if the data has already been collected
                            print("Done")
                            return self.__extract_features(data)
                        elif re.match(re.compile(f"{self.user_id}:"), line):
                            collect = True  # if this is the user id we are looking for, start collecting
                        else:  # if the user_id is different that our
                            continue
                    else:
                        if collect:
                            data.append(line)
        return self.__extract_features(data)

    def compute_similar(self) -> Dict:
        """Parse the file line by line, compute similarity between users, and if high, save the user,
        similarity score and preferred movies"""
        friend_id = None
        data: List[str] = []
        for file_name in [
            os.path.join(DATA_DIR, COMBINED_DATA.format(i + 1)) for i in list(range(4))
        ]:
            with open(file_name) as f:
                for line in f:
                    line = line.strip()
                    if re.match(r"\d+:", line):
                        if friend_id:
                            features = self.__extract_features(
                                data
                            )  # parse lines of data to features
                            similarity = self.similarity(
                                self.my_features, features
                            )  # compute similarity of our
                            # two users
                            self.similar += [
                                {
                                    "user": friend_id,
                                    "similarity": similarity,
                                    "movies": features,
                                }
                            ]  # save the results for the top
                            # n_similar users
                        friend_id = int(re.findall(r"(\d+):", line)[0])
                        if friend_id > LIMIT_USERS:
                            return self.similar
                        data = []
                    else:
                        if friend_id != self.user_id:
                            data.append(line)
        return self.similar

    def recommend(self) -> Counter:
        """Recommend the most popular movies among similar users that were not watched by the user"""
        similar = self.compute_similar()
        weighted_average = {}
        for feature in similar:
            for movie_id, rating in feature["movies"].items():
                if movie_id not in weighted_average:
                    weighted_average[movie_id] = [0] * 3  # numerator, denominator, count
                weighted_average[movie_id][0] += rating * feature["similarity"]
                weighted_average[movie_id][1] += feature["similarity"]
                weighted_average[movie_id][2] += 1
        result = {
            movie_id: value[0] / value[1] if value[0] > 1e-13 else 0
            for movie_id, value in weighted_average.items()
            if value[2] >self.min_common
        }
        return result

    @staticmethod
    def __extract_features(data: List[str]) -> Dict[int, Dict]:
        """Transform a line of text like "movie_id,rating,date" to a dict
        {int(movie_id): {"rating": int(rating),
                        "date": datetime.strptime(date, "%Y-%m-%d")}"""
        data_m = [line.split(",") for line in data]
        return {int(movie_id): int(rating) for movie_id, rating, date in data_m}

    def similarity(
            self, features1: Dict[int, Dict], features2: Dict[int, Dict]
    ) -> float:
        """Compute the similarity based on the correlation between ratings of movies"""
        common = set(features1.keys()).intersection(features2.keys())
        common = [x for x in common if x < LIMIT_MOVIES]
        if len(common) < self.min_common:
            return .0
        val1 = [features1[k] for k in common]
        val2 = [features2[k] for k in common]
        return float((pearsonr(val1, val2)[0] + 1) / 2)


if __name__ == "__main__":
    r = Recommender(user_id=1, n_similar=5, min_common=10)
    my_movies = list(r.my_features.keys())
    print(len(r.my_features))
    train, test = train_test_split(my_movies)
    test_features = {k: v for k, v in r.my_features.items() if k in test}
    r.my_features = {k: v for k, v in r.my_features.items() if k in train}
    print(len(r.my_features))
    recommended = r.recommend()
    print("Similarity: ", r.similarity(test_features, recommended))
    recommended_series = pd.DataFrame(
        recommended.items(), columns=["movie", "recommendation"]
    ).sort_values("recommendation", ascending=False)
    print(recommended_series)
