from typing import Dict, List
from Logic.core.search import SearchEngine
from Logic.core.utility.spell_correction import SpellCorrection
from Logic.core.utility.snippet import Snippet
from Logic.core.indexer.indexes_enum import Indexes, Index_types
import string
import json

try:
    with open('Logic/core/crawler/IMDB_crawled.json', 'r', encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("IMDB_crawled.json not found, initializing an empty list or dict.")
    data = {}
movies_dataset = data
search_engine = SearchEngine('Logic/core')


def correct_text(text: str, all_documents: List[dict]) -> str:
    """
    Correct the give query text, if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text
    all_documents : list of str
        The input documents.

    Returns
    str
        The corrected form of the given text
    """
    all_documents_string = list()
    for movie in all_documents:
        if movie is not None:
            if movie['stars'] is not None:
                all_documents_string.extend(movie['stars'])
            if movie['genres'] is not None:
                all_documents_string.extend(movie['genres'])
            if movie['summaries'] is not None:
                all_documents_string.extend(movie['summaries'])

    for idx, term in enumerate(all_documents_string):
        all_documents_string[idx] = term.translate(str.maketrans('', '', string.punctuation))
    spell_correction_obj = SpellCorrection(all_documents_string)
    text = spell_correction_obj.spell_check(text)
    return text


def search(
    query: str,
    max_result_count: int,
    method: str = "ltn-lnn",
    weights: list = [0.3, 0.3, 0.4],
    should_print=False,
    preferred_genre: str = None,
):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    query:
        The query text

    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    weights:
        The list, containing importance weights in the search result for each of these items:
            Indexes.STARS: weights[0],
            Indexes.GENRES: weights[1],
            Indexes.SUMMARIES: weights[2],

    preferred_genre:
        A list containing preference rates for each genre. If None, the preference rates are equal.
        (You can leave it None for now)

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    weights = {
        Indexes.STARS: weights[0],
        Indexes.GENRES: weights[1],
        Indexes.SUMMARIES: weights[2],
    }
    return search_engine.search(
        query, method, weights, max_results=max_result_count, safe_ranking=True
    )


def get_movie_by_id(id: str, movies_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """
    result = {}
    for movie in movies_dataset:
        if movie['id'] == id:
            result = movie
            break

    result["Image_URL"] = (
        "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"

    )
    result["URL"] = (
        f"https://www.imdb.com/title/{result['id']}"
    )
    return result