FROM gitlab.ilabt.imec.be:4567/ilabt/gpu-docker-stacks/tensorflow-notebook:latest

USER root

RUN apt-get update && apt-get install libgl1-mesa-glx -y 

RUN pip install --upgrade  pip 
RUN pip install albumentations
RUN pip install opencv-python 
RUN pip install matplotlib 
RUN pip install gpxpy 
RUN pip install ultralytics==8.0.117 
RUN pip install plotly 
RUN pip install -U kaleido 
RUN pip install geopandas 
RUN pip install contextily 
RUN pip install folium 
RUN pip install keract

USER jovyan