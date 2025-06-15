from bs4 import BeautifulSoup
import requests
import re
from googlesearch import search
# pip install googlesearch-python

class WikiScraper:

    def __init__(self, url: str, destination="data/wiki/"):
        self.url = url
        self.destination = destination + url[self.url.rindex("/"):].replace(":", "") + ".txt"
        self.paragraphes = []

    def scrap(self):
        print(f"Scrap {self.url}")
        page = requests.get(self.url)
        soup = BeautifulSoup(page.content, 'html.parser')
        nodes = soup.find_all('p')
        for node in nodes:
            paragraphe = node.get_text().strip()
            sentences = paragraphe.split("\n")
            for sentence in sentences:
                if len(sentence) > 1:
                    if sentence[0] != "{":
                        sentence = self.replace(sentence)
                        self.paragraphes.append(sentence)

    def replace(self, s:str) -> str:
        s = re.sub(r"\[\d\]", "", s)
        s = s.replace(" ", " ")
        return s

    def save(self):
        with open(self.destination, "w", encoding="utf-8") as f:
            for paragraphe in self.paragraphes:
                f.write(paragraphe + "\n")

    @staticmethod
    def search_and_scrap(q: str, destination="data/wiki/", num_results=10, lang="fr"):
        results = search(q, lang=lang, num_results=num_results, unique=True)
        for result in results:
            if "wikipedia.org" in result:
                try:
                    scraper = WikiScraper(result, destination=destination)
                    scraper.scrap()
                    scraper.save()
                except Exception as ex:
                    print(f"Error {ex}")




if __name__ == '__main__':
    # url = "https://fr.wikipedia.org/wiki/Grand_mod%C3%A8le_de_langage"
    # scrap = WikiScraper(url)
    # scrap.scrap()
    # scrap.save()
    # url = "https://fr.wikipedia.org/wiki/G%C3%A9n%C3%A9ration_augment%C3%A9e_de_r%C3%A9cup%C3%A9ration"
    # scrap = WikiScraper(url)
    # scrap.scrap()
    # scrap.save()
    # WikiScraper.search_and_scrap("wikipedia large language model", num_results=100)
    WikiScraper.search_and_scrap("wikipedia bâle 3", destination="data/bale3/", num_results=100)

