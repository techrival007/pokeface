import requests
from bs4 import BeautifulSoup
import json
import time
from typing import Dict, List
import os
from urllib.parse import urljoin

def download_image(url: str, save_path: str) -> bool:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading image from {url}: {str(e)}")
        return False

def get_pokemon_data() -> List[Dict]:
    url = "https://pokemondb.net/pokedex/all"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Create images directory if it doesn't exist
    images_dir = 'images'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table with Pokémon data
        table = soup.find('table', {'id': 'pokedex'})
        if not table:
            raise Exception("Could not find Pokémon table")
            
        pokemon_list = []
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                # Get Pokémon number and name
                number = cols[0].text.strip()
                name = cols[1].text.strip()
                
                # Get image URL and download it
                img_element = cols[0].find('img')
                if img_element and img_element.get('src'):
                    img_url = urljoin(url, img_element['src'])
                    img_filename = f"{number}_{name.lower().replace(' ', '_')}.png"
                    img_path = os.path.join(images_dir, img_filename)
                    
                    if download_image(img_url, img_path):
                        print(f"Downloaded image for {name}")
                    else:
                        img_path = None
                else:
                    img_path = None
                
                # Get types
                types = [type_elem.text.strip() for type_elem in cols[2].find_all('a')]
                
                # Get stats
                stats = {
                    'hp': cols[3].text.strip(),
                    'attack': cols[4].text.strip(),
                    'defense': cols[5].text.strip(),
                    'sp_attack': cols[6].text.strip(),
                    'sp_defense': cols[7].text.strip(),
                    'speed': cols[8].text.strip()
                }
                
                pokemon_data = {
                    'number': number,
                    'name': name,
                    'image_path': img_path,
                    'types': types,
                    'stats': stats
                }
                
                pokemon_list.append(pokemon_data)
                print(f"Scraped data for {name}")
                
                # Add a small delay to be respectful to the server
                time.sleep(0.1)
        
        return pokemon_list
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return []

def save_to_json(data: List[Dict], filename: str = 'pokemon_data.json'):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")

def main():
    print("Starting Pokémon data scraping...")
    pokemon_data = get_pokemon_data()
    
    if pokemon_data:
        print(f"Successfully scraped {len(pokemon_data)} Pokémon")
        save_to_json(pokemon_data)
    else:
        print("No data was scraped")

if __name__ == "__main__":
    main() 