import os
import json

import pandas as pd

import config
from map_visualisation import *
from video import *
from osm import *
from google_maps_testing import GoogleMaps


# Load the JSON data from the file
with open(os.path.join(config.PATH, 'metadata_project.json')) as file:
    meta_data = json.load(file)

for course in meta_data["courses"] :
    print(f"Processing course {course}")
    # -----------------------------------------------------------------------------
    # get coordinate points (latitude, longitude) of helicopter and cycling route
    # -----------------------------------------------------------------------------
    print(f"Get coordinate points (latitude, longitude) of helicopter and cycling route")
    print(meta_data["courses"][course])
    course_points_df = extract_points_from_gpx_file(os.path.join(
        config.PATH, 'inputs/gpx-files', meta_data["courses"][course]["gpx_course"]), f"course_route_{course}")
    maxlat = course_points_df['latitude'].max()
    minlat = course_points_df['latitude'].min()
    maxlon = course_points_df['longitude'].max()
    minlon = course_points_df['longitude'].min()
    heli_points_df = extract_points_from_csv_file(os.path.join(
        config.PATH, 'inputs/csv-files', meta_data["courses"][course]["csv_heli"]), f"heli_route_{course}")

    # -----------------------------------------------------------------------------
    # interpolate points
    # -----------------------------------------------------------------------------
    distance = 10
    print(f"Interpolate points of route with a distance of {distance}")
    course_points_interpolated_df = interpolate_points(course_points_df, 10)
    # calculate_max_deviation(course_points_interpolated_df, 10)

    # -----------------------------------------------------------------------------
    # visualize the cycling route and helicopter route on a map in browser
    # -----------------------------------------------------------------------------
    # dfs = [course_points_df, course_points_interpolated_df, heli_points_df]
    # # dfs = [course_points_df, heli_points_df]
    # m = create_map(dfs)
    # display_map(m)

    # -----------------------------------------------------------------------------
    # get the metadata for the coordinate points of the cycling route
    # -----------------------------------------------------------------------------
    # around = 10
    # read_metadata_from_dataframes(course_points_interpolated_df, around)

    # -----------------------------------------------------------------------------
    # take frames of helicopter video (every second)
    # -----------------------------------------------------------------------------
    vid_path = os.path.join(config.PATH, 'inputs/videos', meta_data["courses"][course]["vid"])
    alias = meta_data["courses"][course]["alias"]
    read_frames_of_vid(vid_path, alias, course, 5)
    # detect_bool = True
    # processing_frames_of_vid(vid_path, alias, 1, course, heli_points_df, course_points_interpolated_df, detect_bool)
    # get_maps(vid_path, alias, 1, heli_df=heli_points_df, route_df=course_points_interpolated_df)

    # -----------------------------------------------------------------------------
    # for every type of road marking, extract the OSM coordinates using the tags in the .JSON file
    # and calculate the nearest point of the course and replace it by that point
    # -----------------------------------------------------------------------------
    # for marking in meta_data["markings"] :
    #     print(f"Processing road marking {marking}")
    #     osm_element = list(meta_data["markings"][marking].keys())[0]
    #     osm_tags = meta_data["markings"][marking][osm_element]
    #     # marking_points = get_tag_data_nodes(minlat, maxlat, minlon, maxlon, osm_tags)
    #     marking_points = get_tag_data(course_points_interpolated_df, osm_tags)
    #     print(marking_points)
    #     googlemaps = GoogleMaps(config.GOOGLE_API_KEY)
    #     googlemaps.get_satellite_images(marking_points, marking)
    #     print(f"Done getting Google Maps satellite images for road marking {marking}")

    #     tag_points_of_course = []
    #     point = (0, 0)
    #     for index, row in tag_points.iterrows():
    #         distance = 1000
    #         for index_, row_ in course_points_df.iterrows() :
    #             if haversine(row['latitude'], row['longitude'], row_['latitude'], row_['longitude']) < distance :
    #                 tag_lat = row_['latitude']
    #                 tag_lon = row_['longitude']
    #                 distance = haversine(row['latitude'], row['longitude'], row_['latitude'], row_['longitude'])
    #                 point = (tag_lat, tag_lon)
    #         tag_points_of_course.append(point)
    # tag_points = pd.DataFrame(tag_points_of_course, columns=['latitude', 'longitude'])


    # -----------------------------------------------------------------------------
    # make video with 1 FPS
    # -----------------------------------------------------------------------------
    # demo_video(os.path.join(path, 'Frames_done'), os.path.join(path, 'DemoVideo.mp4'), 1)

