from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = list()
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after 'title/'.
        For example, the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site, or None if an error occurs
        """
        try:
            parts = URL.split('/')
            title_index = parts.index('title')
            if title_index + 1 < len(parts):
                return parts[title_index + 1]
            else:
                print("Error: 'title' segment found, but no ID follows in the URL.")
                return None
        except ValueError:
            print("Error: The URL does not contain the 'title' segment.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        try:
            with open('IMDB_crawled.json', 'w', encoding='utf-8') as file:
                json.dump(self.crawled, file, ensure_ascii=False, indent=4)

            with open('IMDB_not_crawled.json', 'w', encoding='utf-8') as file:
                json.dump(list(self.not_crawled), file, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"An error occurred while writing to JSON file: {e}")

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        try:
            with open('IMDB_crawled.json', 'r') as f:
                self.crawled = json.load(f)
        except FileNotFoundError:
            print("IMDB_crawled.json not found, initializing an empty list or dict.")
            self.crawled = {}

        try:
            with open('IMDB_not_crawled.json', 'r') as f:
                self.not_crawled = json.load(f)
        except FileNotFoundError:
            print("IMDB_not_crawled.json not found, initializing an empty list or dict.")
            self.not_crawled = {}

        # self.added_ids = None

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        try:
            response = get(URL, headers=self.headers)
            if response.status_code == 200:
                return response
            else:
                print(f"Failed to crawl {URL}: Status code {response.status_code}")
                return None
        except Exception as e:
            print(f"An error occurred while crawling {URL}: {e}")
            return None

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        try:
            response = self.crawl(self.top_250_URL)
            if response is None or response.status_code != 200:
                print("Failed to retrieve the Top 250 movies list.")
                return

            soup = BeautifulSoup(response.content, 'html.parser')
            movies = soup.find_all('li', class_='ipc-metadata-list-summary-item')
            self.not_crawled = deque()
            self.added_ids = set()

            for movie in movies:
                link = movie.find('a', href=True)
                movie_id = link['href'].split('/')[2]
                if movie_id not in self.added_ids:
                    self.not_crawled.append(f"https://www.imdb.com/title/{movie_id}/")
                    self.added_ids.add(movie_id)
        except Exception as e:
            print(f"An error occurred while extracting the Top 250 movies: {e}")

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        self.extract_top_250()
        futures = list()
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while crawled_counter < self.crawling_threshold and self.not_crawled:
                if self.not_crawled:
                    URL = self.not_crawled.popleft()
                    future = executor.submit(self.crawl_page_info, URL, crawled_counter)
                    futures.append(future)
                    crawled_counter += 1

                if not self.not_crawled:
                    wait(futures)
                    futures = list()

            if futures:
                wait(futures)

    def crawl_page_info(self, URL, crawler_counter):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        crawler_counter: int
            number of crawled pages
        """
        response = self.crawl(URL)
        if response is None:
            print(f"Failed to crawl page: {URL}")
            return

        soup = BeautifulSoup(response.content, 'html.parser')
        movie = self.get_imdb_instance()

        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = self.get_title(soup)
        movie['first_page_summary'] = self.get_first_page_summary(soup)
        movie['related_links'] = self.get_related_links(soup)
        movie['mpaa'] = self.get_mpaa(soup)
        movie['budget'] = self.get_budget(soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(soup)
        movie['rating'] = self.get_rating(soup)
        movie['directors'] = self.get_director(soup)
        movie['writers'] = self.get_writers(soup)
        movie['stars'] = self.get_stars(soup)
        movie['release_year'] = self.get_release_year(soup)
        movie['genres'] = self.get_genres(soup)
        movie['languages'] = self.get_languages(soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(soup)
        movie['summaries'] = self.get_summaries(URL)
        movie['synopsis'] = self.get_synopsis(URL)
        movie['reviews'] = self.get_reviews_with_scores(URL)

        with self.add_list_lock:
            self.crawled.append(movie)

        related_links = self.get_related_links(soup)
        if related_links:
            with self.add_queue_lock:
                for link in related_links:
                    movie_id = self.get_id_from_URL(link)
                    if movie_id and movie_id not in self.crawled and movie_id not in self.added_ids:
                        self.not_crawled.append(link)
                        self.added_ids.add(movie_id)

        print(crawler_counter, f"Successfully crawled and processed: {URL}")
        return movie

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        # TODO
        movie['title'] = None
        movie['first_page_summary'] = None
        movie['release_year'] = None
        movie['mpaa'] = None
        movie['budget'] = None
        movie['gross_worldwide'] = None
        movie['directors'] = None
        movie['writers'] = None
        movie['stars'] = None
        movie['related_links'] = None
        movie['genres'] = None
        movie['languages'] = None
        movie['countries_of_origin'] = None
        movie['rating'] = None
        movie['summaries'] = None
        movie['synopsis'] = None
        movie['reviews'] = None

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            movie_id = self.get_id_from_URL(url)
            if movie_id:
                return f"https://www.imdb.com/title/{movie_id}/plotsummary"
            else:
                print("Failed to extract movie ID from URL.")
                return None
        except Exception as e:
            print(f"Failed to get summary link: {e}")
            return None

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            movie_id = self.get_id_from_URL(url)
            if movie_id:
                return f"https://www.imdb.com/title/{movie_id}/reviews"
            else:
                print("Failed to extract movie ID from URL.")
                return None
        except Exception as e:
            print(f"Failed to get review link: {e}")
            return None

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            title_tag = soup.find('h1')
            if title_tag:
                return title_tag.text.strip()
            else:
                print("Title tag not found.")
                return None
        except Exception as e:
            print(f"Failed to get title: {e}")
            return None

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            summary_tag = soup.find("span", {"data-testid": "plot-xl"})
            if summary_tag:
                return summary_tag.text.strip()
            else:
                return "Summary not found."
        except Exception as e:
            print(f"Failed to get summary: {e}")
            return "Failed to extract summary."

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            directors = list()
            director_container = soup.find_all('li', {'data-testid': 'title-pc-principal-credit'})
            for container in director_container:
                label = container.find('span', class_='ipc-metadata-list-item__label')
                if label and label.text.strip() == 'Director':
                    director_links = container.find_all('a', class_='ipc-metadata-list-item__list-content-item')
                    for link in director_links:
                        directors.append(link.text.strip())

                label = container.find('a', class_='ipc-metadata-list-item__label')
                if label and label.text.strip() == 'Director':
                    director_links = container.find_all('a', class_='ipc-metadata-list-item__list-content-item')
                    for link in director_links:
                        directors.append(link.text.strip())
            return list(set(directors)) if directors else []
        except Exception as e:
            print(f"Failed to get director: {e}")
            return []

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            stars = []
            stars_container = soup.find_all('li', {'data-testid': 'title-pc-principal-credit'})
            for container in stars_container:
                label = container.find('span', class_='ipc-metadata-list-item__label')
                if label and label.text.strip() == 'Stars':
                    director_links = container.find_all('a', class_='ipc-metadata-list-item__list-content-item')
                    for link in director_links:
                        stars.append(link.text.strip())

                label = container.find('a', class_='ipc-metadata-list-item__label')
                if label and label.text.strip() == 'Stars':
                    director_links = container.find_all('a', class_='ipc-metadata-list-item__list-content-item')
                    for link in director_links:
                        stars.append(link.text.strip())
            return list(set(stars)) if stars else []
        except Exception as e:
            print(f"Failed to get director: {e}")
            return []

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            writers = []
            writer_container = soup.find_all('li', {'data-testid': 'title-pc-principal-credit'})
            for container in writer_container:
                label = container.find('span', class_='ipc-metadata-list-item__label')
                if label and label.text.strip() == 'Writers':
                    director_links = container.find_all('a', class_='ipc-metadata-list-item__list-content-item')
                    for link in director_links:
                        writers.append(link.text.strip())

                label = container.find('a', class_='ipc-metadata-list-item__label')
                if label and label.text.strip() == 'Writers':
                    director_links = container.find_all('a', class_='ipc-metadata-list-item__list-content-item')
                    for link in director_links:
                        writers.append(link.text.strip())
            return list(set(writers)) if writers else []
        except Exception as e:
            print(f"Failed to get director: {e}")
            return []

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            related_links = []
            related_section = soup.find("section", {"data-testid": "MoreLikeThis"})
            if related_section:
                links = related_section.find_all("a", class_="ipc-lockup-overlay ipc-focusable")
                for link in links:
                    href = link.get('href')
                    if href:
                        related_links.append(f"https://www.imdb.com{href}")
            return related_links
        except Exception as e:
            print(f"Failed to get related links: {e}")
            return []

    def get_summaries(self, url):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        summaries = []
        try:
            summaries_url = self.get_summary_link(url)
            if summaries_url:
                summary_response = self.crawl(summaries_url)
                if summary_response and summary_response.status_code == 200:
                    summary_soup = BeautifulSoup(summary_response.content, 'html.parser')
                    summary_sections = summary_soup.find("div", {"data-testid": "sub-section-summaries"})
                    summary_texts = summary_sections.find_all("li")
                    for text in summary_texts:
                        summaries.append(text.find("div", class_="ipc-html-content-inner-div").text.strip())
        except Exception as e:
            print(f"Failed to get summaries: {e}")

        return summaries

    def get_synopsis(self, url):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        synopsis = []
        try:
            synopsis_url = self.get_summary_link(url)
            if synopsis_url:
                synopsis_response = self.crawl(synopsis_url)
                if synopsis_response and synopsis_response.status_code == 200:
                    synopsis_soup = BeautifulSoup(synopsis_response.content, 'html.parser')
                    synopsis_div = synopsis_soup.find_all('div', {"data-testid": "sub-section-synopsis"})
                    if synopsis_div:
                        for div in synopsis_div:
                            synopsis.append(div.text.strip())
                        return synopsis
                    else:
                        print("Synopsis content not found.")
                        return []
                else:
                    print("Failed to retrieve the synopsis page.")
                    return []
            else:
                print("Synopsis url not found.")
                return []
        except Exception as e:
            print(f"Failed to get synopsis: {e}")
            return []

    def get_reviews_with_scores(self, url):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        reviews_with_scores = []
        try:
            reviews_url = self.get_review_link(url)
            if reviews_url:
                response = self.crawl(reviews_url)
                if response and response.status_code == 200:
                    reviews_soup = BeautifulSoup(response.content, 'html.parser')
                    review_divs = reviews_soup.find("div", class_="lister-list")
                    reviews = review_divs.find_all("div", class_="lister-item")
                    for div in reviews:
                        review_text = div.find("div", class_="text show-more__control").text.strip()
                        score = div.find("span", class_="rating-other-user-rating")
                        if score:
                            score_text = score.find("span").text.strip()
                        else:
                            score_text = "N/A"

                        reviews_with_scores.append([review_text, score_text])
                else:
                    print("Failed to retrieve the review page.")
                    return []
            else:
                print("review url not found.")
                return []
        except Exception as e:
            print(f"Failed to get reviews: {e}")
        return reviews_with_scores

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        genres = []
        try:
            genre_list = soup.find('div', class_="ipc-chip-list__scroller")
            if genre_list:
                genre_links = genre_list.find_all("span", class_="ipc-chip__text")
                for genre_link in genre_links:
                    genres.append(genre_link.text.strip())
        except Exception as e:
            print(f"Failed to get genres: {e}")

        return genres

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            rating_tag = soup.find('span', class_='sc-bde20123-1')
            if rating_tag:
                return rating_tag.text.strip()
            else:
                return "Rating not found."
        except Exception as e:
            print(f"Failed to get rating: {e}")
            return "Failed to extract rating."

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            mpaa_tag = soup.find('a', href=lambda href: href and "parentalguide/certificates" in href)
            if mpaa_tag:
                return mpaa_tag.text.strip()
            else:
                return "MPAA rating not found."
        except Exception as e:
            print(f"Failed to get MPAA rating: {e}")
            return "Failed to extract MPAA rating."

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            release_year_tag = soup.find('a', href=lambda href: href and "releaseinfo" in href)
            if release_year_tag:
                return release_year_tag.text.strip()
            else:
                return "Release year not found."
        except Exception as e:
            print(f"Failed to get release year: {e}")
            return "Failed to extract release year."

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            languages = []
            language_section = soup.find('li', {'data-testid': 'title-details-languages'})
            if language_section:
                language_links = language_section.find_all('a', class_='ipc-metadata-list-item__list-content-item')
                for language_link in language_links:
                    languages.append(language_link.text.strip())
            return languages if languages else ["Languages not found."]
        except Exception as e:
            print(f"Failed to get languages: {e}")
            return ["Failed to extract languages."]

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            countries = []
            country_tags = soup.find_all('a', href=lambda href: href and "country_of_origin" in href)
            for tag in country_tags:
                countries.append(tag.text.strip())
            return countries if countries else ["No countries of origin found."]
        except Exception as e:
            print(f"Failed to get countries of origin: {e}")
            return ["Failed to extract countries of origin."]

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            budget_tag = soup.find('li', {'data-testid': 'title-boxoffice-budget'})
            if budget_tag:
                budget = budget_tag.find('span', class_='ipc-metadata-list-item__list-content-item')
                return budget.text.strip() if budget else "Budget not found."
            else:
                return "Budget tag not found."
        except Exception as e:
            print(f"Failed to get budget: {e}")
            return "Failed to extract budget."

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            gross_tag = soup.find('li', {'data-testid': 'title-boxoffice-cumulativeworldwidegross'})
            if gross_tag:
                gross_worldwide = gross_tag.find('span', class_='ipc-metadata-list-item__list-content-item')
                return gross_worldwide.text.strip() if gross_worldwide else "Gross worldwide not found."
            else:
                return "Gross worldwide tag not found."
        except Exception as e:
            print(f"Failed to get gross worldwide: {e}")
            return "Failed to extract gross worldwide."


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=600)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
