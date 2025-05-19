#%%
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://es.restaurantguru.com/Review-Zapopan/reviews?bylang=1'
reviewlist = []

def get_soup(url):
    try:
        page = requests.get('http://localhost:8050/render.html', params={'url': url, 'wait': 2})
        if page.status_code == 200:
            soup = BeautifulSoup(page.text, "html.parser")
            return soup
        else:
            print(f"Error: {page.status_code}")
            return None
    except Exception as e:
        print(f"Error al conectar: {e}")
        return None

def get_reviews(soup):
    if soup:
        container = soup.find('div', {'class': 'scroll-container clear wrapper_reviews'})
        if container:
            # Busca todos los divs con class="o_review" dentro de este contenedor
            reviews = container.find_all('div', {'class': 'o_review'})
            for item in reviews:
                review = {
                    'comment': item.find('span', {'class': 'text_full'}).text.strip() if item.find('span', {'class': 'text_full'}) else 'Sin comentario'
                }
                reviewlist.append(review)
        else:
            print("No se encontró el contenedor con id='comments_conteiner'.")

soup = get_soup(url)
if soup:
    get_reviews(soup)

print(f"Cantidad de reseñas encontradas: {len(reviewlist)}")
for review in reviewlist:
    print(review)

df = pd.DataFrame(reviewlist)
df
# %%
