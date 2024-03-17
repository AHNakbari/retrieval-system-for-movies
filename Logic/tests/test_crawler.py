import json
from typing import List


def check_field_types(json_file_path, expected_fields):
    with open(json_file_path, "r", encoding="utf8") as file:
        data = json.load(file)
    # check len of the data
    assert len(data) > 500, f"Expected at least 1000 movies, but got {len(data)}"

    # check data types
    for movie in data:
        for field, expected_type in expected_fields.items():
            assert (
                field in movie
            ), f'Expected field {field} not found in movie {movie["id"]}'
            if expected_type is not None:
                assert isinstance(
                    movie[field], expected_type
                ), f'Expected field {field} to be of type {expected_type}, but got {type(movie[field])} in movie {movie["id"]}'


expected_fields = {
    "id": str,
    "title": str,
    "first_page_summary": str,
    "release_year": str,
    "mpaa": str,
    "budget": str,
    "gross_worldwide": str,
    "rating": str,
    "directors": List,
    "writers": List,
    "stars": List,
    "related_links": List,
    "genres": List,
    "languages": List,
    "countries_of_origin": List,
    "summaries": List,
    "synopsis": List,
    "reviews": List,
}

json_file_path = "../core/IMDB_crawled.json"
check_field_types(json_file_path, expected_fields)
