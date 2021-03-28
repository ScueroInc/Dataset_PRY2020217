import os
import requests 
from bs4 import BeautifulSoup 
Google_Image = \
    'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'

u_agnt = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive',
} 
data = input('Ingrese el nombre de la comida: ')
Image_Folder = data

def main():
    if not os.path.exists(Image_Folder):
        os.mkdir(Image_Folder)
    download_images()

def download_images():
    
    num_images = int(input('Cantidad de imágenes: '))
    
    print('Buscando imágenes')
    
    search_url = Google_Image + 'q=' + data
    

    response = requests.get(search_url, headers=u_agnt)
    html = response.text 
    b_soup = BeautifulSoup(html, 'html.parser') 
    results = b_soup.findAll('img', {'class': 'rg_i Q4LuWd'})
    count = 0
    imagelinks= []
    for res in results:
        try:
            link = res['data-src']
            imagelinks.append(link)
            count = count + 1
            if (count >= num_images):
                break
            
        except KeyError:
            continue
    
    print(f'Se encontro {len(imagelinks)} imagenes')
    print('Comenzando descarga')

    for i, imagelink in enumerate(imagelinks):
        response = requests.get(imagelink)   
        imagename = Image_Folder + '/' + data + str(i+1) + '.jpg'
        with open(imagename, 'wb') as file:
            file.write(response.content)

    print('Descarga completa')
    

if __name__ == '__main__':
    main()
