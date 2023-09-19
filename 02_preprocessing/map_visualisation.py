import os.path
import subprocess
from math import radians, cos, sin, asin, sqrt, atan2
import folium
import plotly
import xml.etree.ElementTree as ET
import random
import gpxpy
import gpxpy.gpx
import numpy as np
import webcolors
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

path = "/"

# this function extracts DataFrame points from a GPX file
def extract_points_from_gpx_file(path, name) :
    # Parse GPX file
    gpx_file = open(path, 'r')
    gpx = gpxpy.parse(gpx_file)

    # extract latitude and longitude coordinates for each point
    points = [(point.latitude, point.longitude) for track in gpx.tracks for segment in track.segments for point in segment.points]
    points_df = pd.DataFrame(points, columns=['latitude', 'longitude'])
    points_df.name = name
    return points_df

# this function extracts DataFrame points from a CSV file
def extract_points_from_csv_file(path, name) :
    # read CSV file into DataFrame
    df = pd.read_csv(path)

    df['timestamp'] = pd.to_datetime(df['time'])

    # Calculate time difference in seconds from the start point
    start_time = df['timestamp'].iloc[0]
    df['seconds_from_start'] = (df['timestamp'] - start_time).dt.total_seconds()

    # Extract latitude, longitude, and seconds_from_start columns
    points_df = df[['latitude', 'longitude', 'seconds_from_start']]
    points_df.name = name
    return points_df

# using https://nathanrooy.github.io/posts/2016-09-07/haversine-with-python/
def haversine(lat1, lon1, lat2, lon2) :
    R = 6371
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(d_lat/2)**2 + cos(lat1)*cos(lat2)*sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R*c*1000 # returns distance in meters

def calculate_max_deviation(dfs, interval):
    max_dev = 0
    for i in range(1, len(dfs)):
        lat1, lon1 = dfs.iloc[i-1]['latitude'], dfs.iloc[i-1]['longitude']
        lat2, lon2 = dfs.iloc[i]['latitude'], dfs.iloc[i]['longitude']
        dist = haversine(lat1, lon1, lat2, lon2)
        print(dist)
        deviation = abs(interval - dist)
        if deviation > max_dev:
            max_dev = deviation
    print(f"Maximum deviation between interpolated points is : {max_dev}")


# https://stackoverflow.com/questions/51512197/python-equidistant-points-along-a-line-joining-set-of-points
def interpolate_points(points, interval):
    latitude_values = points['latitude'].values
    longitude_values = points['longitude'].values

    # calculate length of whole course
    # for the first point the distance is zero
    length = [0]
    length_val = 0
    for i in range(1, len(latitude_values)) :
        dist = haversine(latitude_values[i-1], longitude_values[i-1], latitude_values[i], longitude_values[i])
        length_val += dist
        length.append(length_val)

    # how much points do we need
    n_points = int(length[-1] / interval)
    # divide length array by last lenght, so every length is between 0 and 1 (with 1 being the total lenght)
    length = np.array(length)
    length = length/length[-1]
    # now calculate interpolation functions
    # interpolation between length and latitude_values -> which latitude value corresponds to a certain length
    # for the first point the distance is zero
    f_lat = interp1d(length, latitude_values)
    # interpolation between length and latitude_values -> which longitude value corresponds to a certain length
    f_lon = interp1d(length, longitude_values)

    # on a full lenght of the route of '1', on what 'length' values those there have to be points
    alpha = np.linspace(0, 1, n_points)
    print(alpha)
    lat_regular = f_lat(alpha)
    lon_regular = f_lon(alpha)

    df = pd.DataFrame({'latitude': lat_regular, 'longitude': lon_regular})
    df.name = "interpolated_points"
    return df





# this function converts DataFrame points to a CSV file
def convert_to_csv_file(points, path_out) :
    points.to_csv(path_out, index=False)

# this function generates a random color in RGB format and convert it to a color name
def get_random_color():
    while True:
        # Generate a random RGB color
        color_hex = '#{:02x}{:02x}{:02x}'.format(*random.sample(range(256), 3))
        try:
            # Try to convert the color to a name
            color_name = webcolors.hex_to_name(color_hex)
            return color_name
        except ValueError:
            # If the conversion fails, try again
            continue


# this function creates a map with the data contained in all the dataframes in the array 'dfs'
def create_map(dfs) :
    # Create a map centered on the middle GPS point
    m = folium.Map(location=[dfs[0]['latitude'][len(dfs[0]['latitude']) // 2], dfs[0]['longitude'][len(dfs[0]['latitude']) // 2]], zoom_start=10)

    # Now go over all the DataFrames
    for df in dfs :
        color = get_random_color()
        print(f"DataFrame {df.name} has color {color}")
        for index, row in df.iterrows() :
            folium.CircleMarker(
                location = (row['latitude'], row['longitude']),
                radius = 2,
                color = color,
                tooltip = str(index),
            ).add_to(m)
    return m

# this function displays the map
def display_map(m) :
    # save the map as an HTML file
    m.save('map.html')

    # Open the HTML file in the default web browser
    subprocess.run(['open', 'map.html'], check=True)


