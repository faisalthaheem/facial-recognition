import logging
import pickle
from bson.binary import Binary
from pymongo import MongoClient
import cv2
from facerecog import FaceRecog
import io
from skimage.transform import resize
import argparse as argparse
from PIL import Image
from pprint import pprint
import numpy as np
import threading
import time
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

# create logger
logger = logging.getLogger('stream.recog')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('stream.recog.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    '[%(levelname)1.1s %(asctime)s] %(message)s', "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

ap = argparse.ArgumentParser()

ap.add_argument("-md", "--model.detector", default="./models/mmod_human_face_detector.dat",
                help="Path to face detection model.")
ap.add_argument("-ms", "--model.shape", default="./models/shape_predictor_68_face_landmarks.dat",
                help="Path to face shape predictor model.")
ap.add_argument("-mr", "--model.recog", default="./models/dlib_face_recognition_resnet_model_v1.dat",
                help="Path to face recognizer model.")

ap.add_argument("-st", "--stream.type", default="live",
                help="set to file to enable pausing when queue hits 5000.")
ap.add_argument("-du", "--mongo.url", default="mongodb://media:27017",
                help="db to store our detections to.")


# example of mjpeg stream from a typical axis camera or esp32
# souphttpsrc location=http://x.y.z.a:8090/mjpg/video.mjpg?timestamp=1564650227443 ! multipartdemux single-stream=true ! image/jpeg,width=320,height=240,framerate=3/1 ! appsink name=sink
# python3 streamrecog.py -cn parking -su "souphttpsrc location=http://x.y.z.a:8090/mjpg/video.mjpg?timestamp=1564650227443 ! multipartdemux single-stream=true ! image/jpeg,width=320,height=240,framerate=3/1 ! appsink name=sink"

#Example of an RTSP source
# rtspsrc location=rtsp://x.y.z.a:5554/camera latency=0 ! rtph264depay ! avdec_h264 ! videorate max-rate=1 ! decodebin ! videoconvert ! video/x-raw, format=BGR ! queue max-size-buffers=1 ! avenc_bmp ! appsink name=sink

#it's important that the sink is appsink to allow this script to pull frames from the pipeline

# ap.add_argument("-su", "--stream.url", default="rtspsrc location=rtsp://localhost:8554/live latency=0 ! rtph264depay ! avdec_h264 ! videorate max-rate=1 ! decodebin ! videoconvert ! video/x-raw, format=BGR ! queue max-size-buffers=1 ! avenc_bmp ! appsink name=sink ",
ap.add_argument("-su", "--stream.url", default="v4l2src device=/dev/video0 ! image/jpeg,framerate=30/1 ! jpegdec ! videoconvert ! avenc_bmp ! appsink name=sink ",
                help="url of stream to consume from")
ap.add_argument("-cn", "--cam.name", required=True,
                help="name of camera to store images from as source.")
ap.add_argument("-iu", "--img.upscale", default=1,
                help="how many times to upscale images to find faces.")

ap.add_argument("-ic", "--img.crop", default="no",
                help="set to 'y' to crop to a ROI and use flag cx,cy,cw,ch.")
ap.add_argument("-cx", "--crop.x", default=0,
                help="the x coordinate to crop from.")
ap.add_argument("-cy", "--crop.y", default=0,
                help="the x coordinate to crop from.")
ap.add_argument("-cw", "--crop.w", default=0,
                help="the x coordinate to crop from.")
ap.add_argument("-ch", "--crop.h", default=0,
                help="the x coordinate to crop from.")



args = vars(ap.parse_args())

os.putenv('GST_DEBUG_DUMP_DIR_DIR', './dots')
os.putenv('GST_DEBUG', '0')

queued_frames = []

frame_grabber = None

GObject.threads_init()
Gst.init(None)


class MainPipeline():
    def __init__(self):
        self.pipeline = None
        self.videosrc = None
        self.videoparse = None
        self.videosink = None
        self.current_buffer = None

    def pull_frame(self, sink):
        sample = sink.emit("pull-sample")
        if sample is not None:

            # caps = sample.get_caps()
            # height = caps.get_structure(0).get_value('height')
            # width = caps.get_structure(0).get_value('width')
            # print(height,width)

            current_buffer = sample.get_buffer()
            current_data = current_buffer.extract_dup(
                0, current_buffer.get_size())

            img = Image.open(io.BytesIO(current_data))
            arr = np.array(img)

            sample = None
            current_data = None
            current_buffer = None

            queued_frames.append((arr, img))

        return Gst.FlowReturn.OK

    def gst_thread(self):
        logger.info("Initializing GST Elements")

        pipeline = args['stream.url']
        self.pipeline = Gst.parse_launch(pipeline)

        # start the video
        logger.info("Setting Pipeline State")
        appsink = self.pipeline.get_by_name("sink")
        appsink.set_property("max-buffers", 1)
        appsink.set_property('emit-signals', True)
        appsink.set_property('sync', False)
        appsink.set_property('wait-on-eos', False)
        appsink.set_property('drop', True)

        appsink.connect('new-sample', self.pull_frame)
        self.pipeline.set_state(Gst.State.PLAYING)

        bus = self.pipeline.get_bus()

        # Parse message
        while True:
            if threading.main_thread().is_alive() is False:
                logger.info("Main thread exited, gst thread exiting")
                return

            message = bus.timed_pop_filtered(10000, Gst.MessageType.ANY)
            if len(queued_frames) > 0:
                pass
            if message:
                if message.type == Gst.MessageType.ERROR:
                    err, debug = message.parse_error()
                    logger.info(("Error received from element %s: %s" % (
                        message.src.get_name(), err)))
                    logger.info(("Debugging information: %s" % debug))
                    break
                elif message.type == Gst.MessageType.EOS:
                    logger.info("End-Of-Stream reached.")
                    break
                elif message.type == Gst.MessageType.STATE_CHANGED:
                    if isinstance(message.src, Gst.Pipeline):
                        old_state, new_state, pending_state = message.parse_state_changed()
                        logger.info(("Pipeline state changed from %s to %s." %
                               (old_state.value_nick, new_state.value_nick)))
                else:
                    logger.info(message)
                    logger.info("Unexpected message received.")

        # Free resources
        self.pipeline.set_state(Gst.State.NULL)


def process_frames(pipeline):

    facerecog = FaceRecog(args["model.detector"],
                          args["model.shape"], args["model.recog"])
    facerecog.initialize()

    # open connection to mongo and start saving
    mongourl = args['mongo.url']
    
    if "mongo.url" in os.environ.keys():
        logger.info('Using mongo.url from environment variable.')
        mongourl = os.environ['mongo.url']

    mongoClient = MongoClient(mongourl)
    mongodb = mongoClient.facials
    collec = mongodb.detections
    frames_processed = 0

    crop_enabled = args["img.crop"] == 'y'
    if crop_enabled:
        crop_x = int(args["crop.x"])
        crop_y = int(args["crop.y"])
        crop_w = int(args["crop.w"])
        crop_h = int(args["crop.h"])

    while True:
        if threading.main_thread().is_alive() is False:
            logger.info("Main thread exited, process_frames exiting")
            return

        qSize = len(queued_frames)
        if qSize > 0:

            if qSize > 10:
                logger.warn("q size[{}]".format(qSize))

                if args['stream.type'] == 'file':
                    if qSize < 100:
                        pipeline.pipeline.set_state(Gst.State.PLAYING)
                    elif qSize > 1000:
                        pipeline.pipeline.set_state(Gst.State.PAUSED)

            arr, img = queued_frames.pop()
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

            if crop_enabled:
                arr = arr[
                    crop_y: crop_y + crop_h,
                    crop_x: crop_x + crop_w
                ]

            face_chips, landmarks, signatures = facerecog.processImage(
                arr, args['img.upscale'])
            frames_processed = frames_processed + 1
            if frames_processed % 10000 == 0:
                logger.info("Frame count {}".format(frames_processed))

            if face_chips is not None and len(face_chips) > 0:

                detections = []

                for i in range(0, len(face_chips)):
                    chip_i = face_chips[i]
                    landmark_i = landmarks[i]
                    signature_i = signatures[i]

                    detection = {
                        "time": time.time(),
                        "chip": Binary(pickle.dumps(chip_i, protocol=2), subtype=128),
                        "landmark": Binary(pickle.dumps(landmark_i, protocol=2), subtype=128),
                        "signature": Binary(pickle.dumps(signature_i, protocol=2), subtype=128),
                        "processed": 0,
                        "cam": args["cam.name"]
                    }

                    detections.append(detection)

                res = collec.insert_many(detections)
                logger.info("inserted [{}] detections.".format(
                    len(res.inserted_ids)))

        else:
            time.sleep(0)


if __name__ == "__main__":

    pipeline = MainPipeline()
    gst_thread = threading.Thread(target=pipeline.gst_thread)
    gst_thread.start()

    frame_thread = threading.Thread(target=process_frames, args=[pipeline])
    frame_thread.start()
    frame_thread.join()

    logger.info("exiting")
