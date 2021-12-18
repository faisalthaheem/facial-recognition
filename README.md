# facial-recognition
Detects, tracks and tags people with randomly assigned names in a db.

![Identified faces in gilette ad](./docs/identified-faces.png?raw=true "Identified faces in gilette ad")

# Quick setup
> Begin with installing pre-requisites for python packages
```bash
sudo apt update && 
DEBIAN_FRONTEND=NON_INTERACTIVE sudo apt install -y \
python3-dev python3-setuptools \
cmake libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
libharfbuzz-dev libfribidi-dev libxcb1-dev
```

> Install gStreamer
```bash
apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
```

> Install python dependencies in a virtualenv
```bash
pip3 install virtualenv &&
virtualenv venv &&
source venv/bin/activate &&
pip3 install --upgrade pip && \
pip3 install dlib pymongo numpy tqdm names scikit_image imutils opencv_python Pillow pgi vext vext.gi
```

> Use of CUDA and cudnn is advised.

# Development and test system
These scripts were developed and tested on ubuntu 20.04. Due to a lot of dependencies between different dependencies ubuntu 20.04 was chosen as the one offering best compatibility out of the box.

# Running
There are a few python scripts, following is an explanation for each

Script | Description
--- | ---
streamrecog.py | The main scripts which can read from a variety of sources with help of gStreamer.
process-db-manual.py | Processes the identified faces and assigns them random names.
chips2png.py | Renders a png with the first few hundred faces from the db.



## Docker compose
It's recommended the docker compose version to be used as it's most straight forward and comes tested. Pre-requisites are docker, docker-compose and nvidia-docker2. Once installed issue the following commands to build the container and run
```bash
su build-docker-images.sh
docker-compose up
```
By default the container will try to use the web-cam. In case you don't have a webcam then please look in the `docker-compose.yaml` file for alternative input sources such as a sample youtube video.

The compose file contains 3 containers, which are
- mongodb listening on port localhost:27017
- mongo-express, a web ui for mongodb listening on http://localhost:8081
- this app



## Viewing identified faces
Assuming the stream has been running and faces have been accumulated in the database, use the following commands to generate a preview
```bash
python3 process-db-manual.py -du mongodb://root:example@localhost:27017/
python3 chips2png.py -du mongodb://root:example@localhost:27017/
```

Following image shows the mongo express with data gatherd on the gilette ad
![Mongo Express with some data](./docs/mongo-express.png?raw=true "Mongo Express with some data")

# Possible applications
This application is a proof of concept that can be extended to use cases of public security by automatically identifying and tagging faces and providing a searchable database.

