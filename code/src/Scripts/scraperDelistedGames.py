import json
import time
import requests
from bs4 import BeautifulSoup

platforms = ["pc","epic-games-store","games-for-windows","gog","microsoft-store","steam","nintendo-switch","nintendo-wii-u","nintendo-wii","nintendo-3ds","nintendo-ds","playstation-5","playstation-4","playstation-3","playstation-2","playstation-vita","playstation-portable","google-stadia"] #.com/all-delisted-xxx-games/
#platforms_xbox =["xbox-series-xs","xbox-one","xbox-360"] #.com/xbox-series-xs-games/

url = "https://www.delistedgames.com/all-delisted-pc-games/"

def site_scraper():
    for platform in platforms:#platform_xbox
        game_scraper("https://www.delistedgames.com/all-delisted-"+platform+"-games/",platform)

def game_scraper(url, platform):
    game = {}
    response = requests.get(url)
    counter = 0
    limit = 3000

    soup = BeautifulSoup(response.text, 'html.parser')

    upper_games = soup.find_all('h4', class_='pt-cv-title')
    for upper_game in upper_games:
        saved_games_data = load()
        ##### utile per fare analisi quando si recupera il lavoro e riposizionarsi sul gioco giusto
        saved_games_names = []
        for saved_game in saved_games_data:
            saved_games_names.append(saved_game["data"]["title"])

        game_info = {}
        game_name = upper_game.contents[0].text
        if game_name not in saved_games_names:
            game_url = upper_game.find('a')['href']
            game_response = requests.get(game_url)
            soup = BeautifulSoup(game_response.text, 'html.parser')
            game_infos_tables = soup.find('table')

        #### Static Hardcoded Version ####
            digital_release = ""
            delisting = ""
            developer = ""
            publisher = ""
            available_on = ""
            links = ""

            index = 0
            if game_infos_tables is not None:
                for row in game_infos_tables.find_all('tr'):
                    columns = row.find_all('td')

                    #Digital Release field
                    if index == 0:
                        digital_release = columns[2].text
                    #Delisting field
                    if index == 1:
                        delisting = columns[1].text
                    #Developer field
                    if index == 2:
                        developer = columns[1].text
                    #Publisher field
                    if index == 3:
                        publisher =columns[1].text
                    #Available on field
                    if index == 4:
                        available_on = columns[1].text
                    #Links field
                    if index == 5:
                        a_elements = columns[1].find_all('a')
                        for a_element in a_elements:
                            href = a_element.get("href")
                            if href is not None:
                                links = links + "-" + href #ahref links
                            else:
                                links = "None"

                    index += 1


            game_info["title"] = game_name
            game_info["platform"] = platform #snellire
            game_info["digital_release"] = digital_release
            game_info["delisting"] = delisting
            game_info["developer"] = developer
            game_info["publisher"] = publisher
            game_info["available_on"] = available_on
            game_info["links"] = links

            textual_infos = soup.find_all('p')
            text_info = []
            for textual_info in textual_infos:
                text_info.append(textual_info.text)

            game_info["info"] = text_info

            ##### save #####
            if(len(saved_games_data) > 0):
                last_game_index = saved_games_data[-1]["id"]
            else:
                last_game_index = -1
            game["id"] = last_game_index + 1
            game["data"] = game_info
            saved_games_data.append(game)
            save(saved_games_data)
            ################

        else:
            continue

        if counter > limit:
            break
        counter += 1

        time.sleep(10)

def save(new_data):
    with open("delistedgames.json", "w") as outfile:
        json.dump(new_data, outfile, indent=4, ensure_ascii=False)

def load():
    with open("delistedgames.json", "r") as file:
        try:
            data = json.load(file)
        except Exception as e:
            print(e)
            data = []
    return data

site_scraper()