import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

import config

tags = []


# using https://towardsdatascience.com/loading-data-from-openstreetmap-with-python-and-the-overpass-api-513882a27fd0
# def read_metadata_from_dataframes(points, around):
#     for i in range(1, len(points)):
#         data = query(points.iloc[i][0], points.iloc[i][1], around)
#         for node in data['elements']:
#             # for every node that contains a tag, print the tag
#             if 'tags' in node:
#                 print(f"Node ID: {node['id']}, Lat: {node['lat']}, Long: {node['lon']}, Tag: {node['tags']}")
#                 for key in node['tags']:
#                     if key not in tags:
#                         tags.append(key)
#             else:
#                 print(f"Node ID: {node['id']}, Lat: {node['lat']}, Long: {node['lon']}")
#             print("---")
#     print("Tags: ")
#     print(tags)


# get query for Overpass API
def get_query_tags(tags):
    query_string = ""
    for tag in tags :
        query_string += f"['{tag}']"
    return query_string


# to get the locations of all the nodes with a 'tag' in a bounding box region ('minlat', 'maxlat', 'minlon', 'maxlon')
def get_tag_data_nodes(minlat, maxlat, minlon, maxlon, tag):
    overpass_query = f"""[out:json];
                     node[{tag}]({minlat},{minlon},{maxlat},{maxlon});
                     out;"""
    response = requests.get(config.OVERPASS_URL, params={'data': overpass_query})

    # Extract the crosswalk locations from the API response and return the datapoints as a DataFrame
    points = []
    for node in response.json()['elements']:
        lat = node['lat']
        lon = node['lon']
        point = (lat, lon)
        points.append(point)
    points_df = pd.DataFrame(points, columns=['latitude', 'longitude'])
    return points_df

# to get latitude and longitude from a node id
def get_data_from_node_id(node_id) :
    overpass_query = f"""[out:json];
                         node({node_id});
                         out;"""
    response = requests.get(config.OVERPASS_URL, params={'data': overpass_query})
    node_data = response.json()['elements'][0]  # Retrieve the first element (node)
    lat = node_data['lat']
    lon = node_data['lon']
    return lat, lon


def get_tag_data(df, tags):
    points = []
    tags_query = get_query_tags(tags)

    for index, row in df.iterrows() :
        lat = row['latitude']
        lon = row['longitude']

        # take all the points in a circle of 5 meters
        overpass_query = f"""[out:json];
                        node{tags_query}(around: 5, {lat},{lon});
                        out;"""
        response = requests.get(config.OVERPASS_URL, params={'data': overpass_query})

        count_tag = 0
        print(response.json()['elements'])
        if response.json()['elements'] is not None:
            for tag in tags:
                if 'tags' in response.json()['elements'] and tag in response.json()['elements']['tags']:
                    count_tag += 1
                    print(count_tag)

        if count_tag == len(tags) :
            point = (lat, lon)
            points.append(point)
            if len(points) == 1:
                return pd.DataFrame(points, columns=['latitude', 'longitude'])

    return pd.DataFrame(points, columns=['latitude', 'longitude'])

    # Extract the crosswalk locations from the API response and return the datapoints as a DataFrame
    # points = []
    # for item in response.json()['elements']:
    #     # only need 1 node
    #     node_id = item['nodes'][0]
    #     lat, lon = get_data_from_node_id(node_id)
    #     point = (lat, lon)
    #     if len(points) == 50 :
    #         points_df = pd.DataFrame(points, columns=['latitude', 'longitude'])
    #         return points_df
    #     points.append(point)
    # points_df = pd.DataFrame(points, columns=['latitude', 'longitude'])
    # return points_df

def get_street_from_node(lat, lon) :
    return None


def get_nodes_from_way_id(way_id):
    overpass_query = f"""[out:json];
                        way({way_id});
                        node(w);
                        out meta;"""
    response = requests.get(config.OVERPASS_URL, params={'data': overpass_query})
    node_data = response.json()['elements'][0]  # Retrieve the first element (node)
    lat = node_data['lat']
    lon = node_data['lon']
    return lat, lon

# processing frames of video, YOLO model on the individual frames of the video
def get_maps(path, alias, time_interval, heli_df, route_df) :
    # create video capture object
    cap = cv2.VideoCapture(path)

    # set frame rate of video, property of the video itself
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video has a fps of {fps} frames per second. ")

    # set time interval (in seconds) for frame capture
    interval = time_interval

    # calculate frame interval based on fps and time interval
    # frame interval in #frames
    frame_interval = int(fps * interval)

    # initialize frame counter
    frame_count = 0

    mapbox_route = go.Scattermapbox(
        lon=route_df['longitude'],
        lat=route_df['latitude'],
        name="Route points",
        mode="markers",
        marker=dict(
            color='red',
            size=8,
        ),
    )

    # loop over frames in video
    while cap.isOpened():
        # read next frame from video
        ret, frame = cap.read()

        # check if frame was read successfully
        if not ret:
            break

        # check if it's time to capture a frame
        if frame_count % frame_interval == 0:
            seconds = frame_count / fps

            # store frames in right directory
            # frame_name = f"{alias}_frame{frame_count}.jpg"
            # print(f"Writing frame {frame_name}, after {seconds // 60} min and {seconds % 60} seconds.")
            # cv2.imwrite(os.path.join(output_dir, frame_name), frame)

            # creating map to display next to frame
            # if no match with seconds, use previous heli points
            if seconds in heli_df['seconds_from_start'].values:
                filtered_df = heli_df[heli_df['seconds_from_start'] == seconds]

            mapbox_heli_other = go.Scattermapbox(
                lon=heli_df['longitude'],
                lat=heli_df['latitude'],
                mode='markers',
                marker=dict(
                    color='blue',
                    size=8,
                ),
                name="Other Helicopter points"
            )

            mapbox_heli_now = go.Scattermapbox(
                lon=filtered_df['longitude'],
                lat=filtered_df['latitude'],
                mode='markers',
                marker=dict(
                    color='yellow',
                    size=8,
                ),
                name="Live Helicopter point(s)"
            )

            # Set up the layout
            layout = go.Layout(
                width=1200,
                height=900,
                mapbox=dict(
                    style="open-street-map",
                    zoom=16,
                    center=dict(
                        lon=filtered_df.iloc[0]['longitude'],
                        lat=filtered_df.iloc[0]['latitude']
                    )
                )
            )

            data = [mapbox_route, mapbox_heli_other, mapbox_heli_now]
            # Create an empty scattermapbox trace
            empty_trace = go.Scattermapbox()

            # fig = go.Figure(data=data, layout=layout)
            fig = go.Figure(data=empty_trace, layout=layout)
            output_map_image = os.path.join(os.getcwd(), "osm_maps", alias, f"map_{seconds}.jpg")
            pio.write_image(fig, output_map_image, format="jpg")

            # Convert the figure to an image in memory
            image_bytes = pio.to_image(fig, format='jpg')
            
            # Convert the image bytes to a NumPy array
            np_arr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode the NumPy array as an image using OpenCV
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Plot the image
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')  # Optional: Turn off axes
            plt.show()


        # increment frame counter
        frame_count += 1

    # release video capture object
    cap.release()

def map_to_mask(dir) :
    for map in os.listdir(dir) :
        map_img = cv2.imread(os.path.join(dir, map))

        
        cv2.imshow("Map image", map_img)
        cv2.waitKey()
        # Convert the image to grayscale
        gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Original Image", gray)
        cv2.waitKey()
        # Apply the threshold operation (inverted)
        _, thresholded = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        masked_map_dir = os.path.join(os.getcwd(), "osm_maps", "2020_KBK_masked_maps", map)
        cv2.imwrite(masked_map_dir, thresholded)
        cv2.imshow("Thresholded Image", thresholded)
        cv2.waitKey()
        cv2.destroyAllWindows()

# osm_maps_dir = os.path.join(os.getcwd(), "osm_maps", "2020_KBK")
# map_to_mask(osm_maps_dir)