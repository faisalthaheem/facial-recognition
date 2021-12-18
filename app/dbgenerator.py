import argparse
import sys
import os
import shutil as sh
import logging
from tqdm import tqdm
from facerecog import FaceRecog
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-sd", "--source.dir", required=True,
                help="Directory containing files to compute signatures of.")
ap.add_argument("-dd", "--database.dir", required=True,
                help="Directory to store signatures in.")

ap.add_argument("-md", "--model.detector", default="../models/mmod_human_face_detector.dat",
                help="Path to face detection model.")
ap.add_argument("-ms", "--model.shape", default="../models/shape_predictor_68_face_landmarks.dat",
                help="Path to face shape predictor model.")
ap.add_argument("-mr", "--model.recog", default="../models/dlib_face_recognition_resnet_model_v1.dat",
                help="Path to face recognizer model.")
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG)
logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("dbgenerator.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


# verify the directories exist
if not os.path.exists(args["source.dir"]) or not os.path.isdir(args["source.dir"]):
    logging.error("%s is not a directory or does not exist." %
                  args["source.dir"])
    sys.exit()

if not os.path.exists(args["database.dir"]) or not os.path.isdir(args["database.dir"]):
    logging.error("%s is not a directory or does not exist." %
                  args["database.dir"])
    sys.exit()

facerecog = FaceRecog(args["model.detector"],
                      args["model.shape"], args["model.recog"])
facerecog.initialize()

logging.info("Processing files in " + args["source.dir"])
for root, dirs, files in os.walk(args["source.dir"]):

    totalFiles = len(files)
    logging.info("[{}] files to process".format(totalFiles))

    for i in tqdm(range(0, totalFiles)):
        fileName = files[i]
        srcPath = os.path.join(root, fileName)
        dstPath = os.path.join(args["database.dir"], fileName + ".sig")

        face_chips, landmarks, signatures = facerecog.getSignatures(srcPath)
        #[print(signature) for signature in signatures]
        for i in range(0, len(signatures)):
            sig_i = signatures[i]
            np.savetxt(dstPath, sig_i)


logging.info("Done..")
