 


# only read in the frames of the video, no processing involved
def read_frames_of_vid(path, alias, course, time_interval) :
    print(f"Reading video frames of {course} to store in directory.")

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

    # output directory for the frames
    output_dir = os.path.join(config.PATH, 'inputs', 'frames-from-vid', alias)

    # create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loop over frames in video
    while cap.isOpened():
        # read next frame from video
        ret, frame = cap.read()
        print(frame.shape)

        # check if frame was read successfully
        if not ret:
            break

        # check if it's time to capture a frame
        if frame_count % frame_interval == 0:
            seconds = frame_count / fps

            # store frames in right directory
            frame_name = f"{alias}_frame{frame_count}.jpg"
            print(f"Writing frame {frame_name}, after {seconds // 60} min and {seconds % 60} seconds.")
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)

        # increment frame counter
        frame_count += 1
    


# processing frames of video, YOLO model on the individual frames of the video
def processing_frames_of_vid(path, alias, time_interval, course, heli_df, route_df, detect_bool) :
    print(f"Processing video frames of {course}")
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

    # load custom YOLO model
    model = YOLO("v4.pt")

    # output directory for the frames
    output_dir = os.path.join(config.PATH, 'inputs', 'frames-from-vid', alias)

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

            filtered_df = filtered_df[0]

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
                title="VENTOUX",
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

            fig = go.Figure(data=data, layout=layout)
            output_map_image = os.path.join(os.getcwd(), "inputs", "map", alias, f"map_{seconds}.jpg")
            pio.write_image(fig, output_map_image, format="jpg")

            # custom YOLO model on frame
            if detect_bool :
                results = model(frame)
                frame = results[0].plot()
                # result = results[0]
                # bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
                # print(result.boxes.data.cpu())
                # classes = np.array(result.boxes.cls.cpu(), dtype="int")
                # for cls, bbox in zip(classes, bboxes) :
                #     (x1, y1, x2, y2) = bbox
                #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #     cv2.putText(frame, str(cls), (x1, y1-5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


            # put map and frame with detection together in 1 image
            map_image = Image.open(output_map_image)

            # Convert video frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a PIL Image from the video frame
            frame_image = Image.fromarray(frame_rgb)

            # Combine the map image and video frame side by side
            combined_image = Image.new('RGB', (map_image.width + frame_image.width, map_image.height))
            combined_image.paste(map_image, (0, 0))
            combined_image.paste(frame_image, (map_image.width, 0))

            # Specify the output image file path for the combined image
            if detect_bool :
                output_combined_map_image = os.path.join(os.getcwd(), "outputs", "demo-with-detection", alias, f"combined_map_frame_{seconds}.jpg")
            else :
                output_combined_map_image = os.path.join(os.getcwd(), "outputs", "demo", alias, f"combined_map_frame_{seconds}.jpg")

            # Save the combined image
            combined_image.save(output_combined_map_image)

        # increment frame counter
        frame_count += 1

    # release video capture object
    cap.release()
    if detect_bool :
        output_dir_combined_map_image = os.path.join(os.getcwd(), "outputs", "demo-with-detection", alias)
        demo_path = os.path.join(output_dir_combined_map_image, f"demo_with_detection_{alias}.mp4")
    else :
        output_dir_combined_map_image = os.path.join(os.getcwd(), "outputs", "demo", alias)
        demo_path = os.path.join(output_dir_combined_map_image, f"demo_{alias}.mp4")
    make_demo_video(output_dir_combined_map_image, demo_path, fps)


def put_text(text, image, text_x, text_y):
    new_image = cv2.putText(
        img=image,
        text=text,
        org=(text_x, text_y),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=image.shape[1] / 1080,
        color=(255, 0, 0),
        thickness=2
    )
    return new_image

def make_demo_video(image_dir, video_path, fps):
    images = [img for img in os.listdir(image_dir) if img.endswith('.jpg')]
    sorted_images = sorted(images, key=lambda x: int(''.join(str(char) for char in x if char.isdigit())))

    # read in first frame to get height and width
    first_frame = cv2.imread(os.path.join(image_dir, sorted_images[0]))
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in sorted_images :
        frame = cv2.imread(os.path.join(image_dir, image))
        video.write(frame)

    print(f"Demo is ready and can be found at: '{video_path}'")
    cv2.destroyAllWindows()
    video.release()

# def demo_video_model(video_path) :
#     model = YOLO("best_v5.pt")
#     model.predict(source=video_path, show=True, save=True, show_labels=True, show_conf=True, conf=0.5, save_txt=False,
#                   save_crop=False, line_width=2, box=True, visualize=False, stream=True)
    

def demo_video_model(video_path) :
    # Load the YOLOv8 model
    model = YOLO('best_v5.pt')

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(frame, show=True, device="mps")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()




# Set the environment variable to enable MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
video_path = os.path.join(os.getcwd(), "inputs", "videos", "2023-Ronde-van-Vlaanderen-1.mp4")
demo_video_model(video_path)

# fps = 1
# output_dir_combined_map_image = os.path.join(os.getcwd(), "outputs", "demo-with-detection", "2023_RVV")
# demo_path = os.path.join(output_dir_combined_map_image, f"demo_2023_RVV.mp4")
# make_demo_video(output_dir_combined_map_image, demo_path, fps)

# frames_from_vid_dir = os.path.join(os.getcwd(), "inputs", "frames-from-vid", "2020_KBK")
# perspective_transform(frames_from_vid_dir)