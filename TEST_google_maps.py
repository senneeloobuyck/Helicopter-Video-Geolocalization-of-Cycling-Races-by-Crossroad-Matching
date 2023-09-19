import requests
import os
import urllib.request
import googlemaps
import config

class GoogleMaps() :

    def __init__(self, key) :
        self.key = key


    def get_satellite_images(self, dataframe, marking) :
        for index, data in dataframe.iterrows() :
            map_client = googlemaps.Client(self.key)
            lat = data['latitude']
            lon = data['longitude']

            os.makedirs(os.path.join(config.PATH, "satellite-images"), exist_ok=True)
            f = open(os.path.join(config.PATH, "satellite-images", f"{marking}-{lat}-{lon}.jpg"), 'wb')
            for chunk in map_client.static_map(size=(400, 400), center=(lat, lon), zoom=21, maptype='satellite'):
                if chunk:
                    f.write(chunk)
            f.close()

        # base_api_static_maps = 'https://maps.googleapis.com/maps/api/staticmap'
        # request_url = f'{base_api_static_maps}?center={lat},{lon}&zoom=15&size=400x400&maptype=satellite&key={key}'
        # print(request_url)
        # urllib.request.urlretrieve(request_url, os.path.join(path, "Found_satelite_photos_with_crossing"))

    # def get_45_degree_photo(self, latitude, longitude):
    #     base_url = "https://maps.googleapis.com/maps/api/streetview"
    #     heading = 45  # Specify the desired heading (in degrees)
    #
    #     # Create the request URL with the necessary parameters
    #     request_url = f"{base_url}?location={latitude},{longitude}&heading={heading}&key={self.key}&size=640x640"
    #
    #     # Send the request and retrieve the image
    #     response = requests.get(request_url)
    #
    #     photo_path = os.path.join(os.getcwd(), "45_degrees", f"lat:{latitude}_lon:{longitude}.jpg")
    #     if response.status_code == 200:
    #         with open(photo_path, "wb") as f:
    #             f.write(response.content)
    #         print("Photo saved successfully.")
    #     else:
    #         print("Error retrieving the photo.")
    #         print(response.content)


googlemaps = GoogleMaps(config.GOOGLE_API_KEY)
# googlemaps.get_45_degree_photo(50.790321, 3.560127)