#FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN DEBIAN_FRONTEND=NONINTERACTIVE apt update && \
    DEBIAN_FRONTEND=NONINTERACTIVE apt install -y python3 python3-pip \
    libglib2.0-0 libglib2.0-dev libglib2.0-dev-bin \
    libgirepository-1.0-1 gobject-introspection libgirepository1.0-dev gir1.2-gstreamer-1.0 \
    python3-dev python3-setuptools \
    cmake libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev libxcb1-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl \
    gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

RUN pip3 install --upgrade pip && \
    pip3 install dlib pymongo numpy tqdm names scikit_image imutils opencv_python Pillow pgi

ADD . /app
WORKDIR /app

CMD ["python3","streamrecog.py","-cn","webcam"]