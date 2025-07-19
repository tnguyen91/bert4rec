from pathlib import Path
import pandas as pd
import kagglehub
import os
os.environ["KAGGLEHUB_CACHE"] = "./data"
from typing import Tuple

def find_file(filename: str, search_dir: str = "./data") -> Path:
    matches = list(Path(search_dir).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found in {search_dir}")
    return matches[0]

def filter_hentai(ratings: pd.DataFrame, anime: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = ~anime.apply(lambda row: row.astype(str).str.contains('Hentai', case=False, na=False)).any(axis=1)
    anime_clean = anime[mask]
    ratings_clean = ratings[ratings['anime_id'].isin(anime_clean['anime_id'])]
    return ratings_clean, anime_clean

def load_dataset(data_dir: str = "./data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        kagglehub.dataset_download("bsurya27/myanimelists-anime-and-user-anime-interactions")
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")

    rating_path = find_file("User-AnimeReview.csv", search_dir=data_dir)
    anime_path = find_file("Anime.csv", search_dir=data_dir)

    try:
        ratings = pd.read_csv(rating_path)
        anime = pd.read_csv(anime_path)
    except Exception as e:
        raise ValueError(f"Failed to load CSVs: {e}")

    return filter_hentai(ratings, anime)
