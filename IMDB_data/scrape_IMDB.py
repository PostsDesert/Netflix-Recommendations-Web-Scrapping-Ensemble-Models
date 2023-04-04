import csv
import pandas as pd
import requests
from requests_html import HTMLSession
import html
import json
import re
import time
import random

COMPLETED_MOVIE_FILE = 'IMDB_data/completed_movies.csv'
ERROR_MOVIE_FILE = 'IMDB_data/error_movies.csv'

headers = {
    "Accept": "application/json, text/plain, */*",
    # Use Safari macOS user agent
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "Referer": "https://www.imdb.com/"
}

session = HTMLSession()

def get_IMDB_data_by_movie(movie):
    url = f"https://www.imdb.com/find?q={movie['name']}+{movie['year']}&ref_=nv_sr_sm"
    r = session.get(url, headers=headers)
    results = r.html.xpath("//section[@data-testid='find-results-section-title']/div/ul/li")
    # get all list items

    page_url = None
    for result in results:
        name = result.text.replace('\n', ' ')
        path_url = result.find('a', first='true').attrs['href']

        print(name)
        print(f"{movie['name'].lower()} === {name.split(' ')[0].lower()}")
        print("Podcast" not in name, "Music Video" not in name, movie['name'].lower().startswith(name.split(" ")[0].lower()))
        if "Podcast" not in name and "Music Video" not in name and movie['name'].lower().startswith(name.split(" ")[0].lower()):
            print(f"Found match for: {movie['name']} Actual name: {name}")
            page_url = f"https://www.imdb.com{path_url}"
            break

    if page_url is not None:
        print(f"Found page url: {page_url}")
        r = session.get(page_url, headers=headers)
        result = r.html.xpath("//script[@type='application/ld+json']")[0].text
        # result = """{"@context":"https://schema.org","@type":"Movie","url":"/title/tt0119448/","name":"Karakter","alternateName":"Character","image":"https://m.media-amazon.com/images/M/MV5BNzUzMzk2Y2YtNTNjNC00YzVmLWEzMWEtYWI0MWVjNWQ0MDNjXkEyXkFqcGdeQXVyNjMwMjk0MTQ@._V1_.jpg","description":"Jacob Katadreuffe lives mute with his mother, has no contact with his father who only works against him and wants to become a lawyer, at all costs.","review":{"@type":"Review","itemReviewed":{"@type":"CreativeWork","url":"/title/tt0119448/"},"author":{"@type":"Person","name":"khatcher-2"},"dateCreated":"2004-04-10","inLanguage":"English","name":"Beautifully filmed 1920's "Rotterdam"","reviewBody":"It is not too frequent that we get Dutch programmes of films or TV-minis in this corner of Europe, and when they do appear it is thanks to the regional Basque TV Station `EITB'. Indeed over two years has passed since seeing the excellent mini `Charlotte Sophie Bentinck' (1996) (qv) and seeing the very interesting `Karakter' recently. \n\nSet in the 1920's this film has excellent mise-en-scéne wonderfully photographed, mostly in Holland and Belgium, but with some scenes shot in Wroclaw, Poland, with street-cars of the times, in which the darkened almost greyish brickwork of the tenement buildings and the industrial port areas takes on an intense protagonism in the film's development. Palais van Boem's musical contribution is mostly just right, though at times seemed to be a little boorish.\n\nA young, illegitimate boy grows up with his unmarried mother, whilst the father, Dreverhaven, continuously appeals to her to marry him, but always rejected. However, the father seems to do everything possible to disrupt the young man's life, as his mother becomes more and more detached and uncaring. It would seem that Dreverhaven is playing out a real-life game of chess around his son Jacob, as if trying to corner him into submission and apathy, but which the young man manages to survive. The psychological impression is that one or the other would undo his `bitter foe', but that despite the father's vast fortune and power the struggle of will would rebound against him.But as the Dutch saying goes: De één zijn dood, is de ander zijn brood'\n\nThis is no `thriller' in the ordinary sense, more a psychological suspense which requires attention throughout. The acting is magnificent: both Fedja van Huêt and Jan Decleir play out their parts with just the right touch, especially Decleir, and Lou Landré as Rentenstein is almost spellbinding, not to be missed.\n\nHere is another example of the unarguable fact: here in Europe we make cinema, not blockbuster box-office hits.","reviewRating":{"@type":"Rating","worstRating":1,"bestRating":10,"ratingValue":7}},"aggregateRating":{"@type":"AggregateRating","ratingCount":11026,"bestRating":10,"worstRating":1,"ratingValue":7.7},"contentRating":"R","genre":["Crime","Drama","Mystery"],"datePublished":"1998-03-27","keywords":"gunfight,street shootout,police shootout,fiance fiancee relationship,police","actor":[{"@type":"Person","url":"/name/nm0208798/","name":"Pavlik Jansen op de Haar"},{"@type":"Person","url":"/name/nm0213912/","name":"Jan Decleir"},{"@type":"Person","url":"/name/nm0404806/","name":"Fedja van Huêt"}],"director":[{"@type":"Person","url":"/name/nm0226016/","name":"Mike van Diem"}],"creator":[{"@type":"Organization","url":"/company/co0000316/"},{"@type":"Organization","url":"/company/co0032474/"},{"@type":"Person","url":"/name/nm0096205/","name":"Ferdinand Bordewijk"},{"@type":"Person","url":"/name/nm0311597/","name":"Laurens Geels"},{"@type":"Person","url":"/name/nm0226016/","name":"Mike van Diem"}],"duration":"PT2H2M"}"""

        fixed_json = fix_json(result).encode('utf-8')

        print(fixed_json)

        result = json.loads(fixed_json)

        result['original_info'] = {
            'name': movie['name'],
            'year': movie['year'],
            'id': f'{movie["name"].replace("/", "|")}_{movie["year"]}'
        }

        return json.dumps(result, indent=4, sort_keys=True, ensure_ascii=False)
    else:
        print(f"ERROR: Could not find match for: {movie['name']} at url: {url}")
        # write error string to file
        with open("IMDB_data/error.txt", "a") as f:
            f.write(f"ERROR: Could not find match for: {movie['name']} at url: {url}\n")
        write_movie_to_file(ERROR_MOVIE_FILE, movie)
        return None

def fix_json(json_str):
    # Unescape all HTML entities
    decode_str = html.unescape(json_str)

    # remove all new lines
    json_str = ''.join(json_str.splitlines())

    fields_need_linting = [
                            [r'"name":"', '(?:","|"})'],
                            [r'"description":"', '","'],
                            [r'"reviewBody":"', '(","reviewRating":|"},")'],
                            [r'"trailer":"', r'",'],
    ]

    # change all double quotes in the fields to double quotes with escape character
    output_str = decode_str
    for field in fields_need_linting:
        pattern = r'(?<=' + field[0] + r')' + r'.*?' + r'(?=' + field[1] + r')'
        output_str = re.sub(pattern, lambda x: x.group(0).replace('"', '\\"'), output_str)

    return output_str

def read_movies_from_csv(file_path):
    movies = []
    with open(file_path, encoding = "ISO-8859-1") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
                movies.append({"name": row[2], "year": row[1]})
    return movies

def write_IMDB_data_to_file(movie, IMDB_data):
    # we need this tag to match the movie with the IMDB data
    with open(f'IMDB_data/data/{movie["name"].replace("/", "|")}_{movie["year"]}.json', 'w') as outfile:
        outfile.write(IMDB_data)

def write_movie_to_file(file, movie):
    with open(file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['index', 'year', 'name'])
        writer.writerow({'index': '', 'year': movie['year'], 'name': movie['name']})


def main():
    movies = read_movies_from_csv('prize_dataset/movie_titles.csv')

    completed_list = read_movies_from_csv(COMPLETED_MOVIE_FILE)
    completed_list = [movie['name'] for movie in completed_list]
    error_list = read_movies_from_csv(ERROR_MOVIE_FILE)
    error_list = [movie['name'] for movie in error_list]
    dont_search_list = completed_list + error_list
    
    
    IMDB_data = []
    for movie in movies:
        if movie['name'] in dont_search_list:
            print(f"Skipping movie: {movie['name']} ({movie['year']})")
            continue
        
        print("Searching for movie: " + movie['name'] + " (" + movie['year'] + ")")
        IMDB_data = get_IMDB_data_by_movie(movie)
        
        if IMDB_data:
            write_IMDB_data_to_file(movie, IMDB_data)
            write_movie_to_file(COMPLETED_MOVIE_FILE, movie)
        
        # # randmize the time interval between each request
        # time.sleep(random.randint(1, 7))

if __name__ == '__main__':
    main()