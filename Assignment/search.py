#!/bin/python3.13
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup


def get_search_results(query):
    """ Code to get search results from DuckDuckGo. """
    with DDGS() as ddgs:
        results = ddgs.text(
            query = query,
            region   = "wt-wt",
            safesearch = "moderate",
            max_results = 10,
        )
    return results

def get_text_from_url(url):
    """ Code to get text from webpages. """
    response = requests.get(url, headers = {'User-agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(response.text, "html.parser")
    body_content = soup.find("div", id="bodyContent")
    return body_content.get_text() if body_content else None

def main():
    # Get search results based on the query.
    search_results = get_search_results("Andor Season 3 wikipedia")
    print(search_results)
    print()

    # Get the wikipedia result as an example.
    relevant_page = [res["href"] for res in search_results if "wikipedia" in res["href"]][0]
    print(relevant_page)

    # Get HTML from wikipedia.
    web_page = get_text_from_url(relevant_page)
    if web_page is not None:
        print(web_page)


if __name__ == "__main__":
    main()
