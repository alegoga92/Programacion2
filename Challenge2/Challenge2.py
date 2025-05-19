#%%
''
import requests
from bs4 import BeautifulSoup

# Paso 1: Obtener el contenido HTML de la página
url = 'https://es.restaurantguru.com/Review-Zapopan/reviews?bylang=1'
reviewlist =[]

def get_soup(url):
    page = requests.get('http://localhost:8050/render.html', params={'url': url, 'wait': 2})
    soup = BeautifulSoup(page.text, "html.parser")
    return soup

def get_reviews(soup):
    reviews = soup.find_all('div', {'class': 'o_review'})
    for item in reviews:
        review = {
        'comment': item.find('span', {'class': 'text_full'}).text,
        }
    reviewlist.append(review)

print(len(reviewlist))
# %%
