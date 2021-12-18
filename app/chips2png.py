import time
from bson.binary import Binary
import pickle
from pymongo import MongoClient
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-du", "--mongo.url", default="mongodb://mongo:27017",
                help="db to store our detections to.")
args = vars(ap.parse_args())

mongoClient = MongoClient(args['mongo.url'])
mongodb = mongoClient.facials
detections = mongodb.detections
persons = mongodb.persons

mem_persons = {}
cursor = persons.find({})
for doc in cursor:
    mem_persons[doc['_id']] = doc['full_name']


time_start = time.time()

COLS_PER_ROW = 20
ROWS_PER_IMG = 50

CHIP_WIDTH = 150
CHIP_HEIGHT = 150

PNG_WIDTH = COLS_PER_ROW * CHIP_WIDTH
PNG_HEIGHT = ROWS_PER_IMG * CHIP_HEIGHT
PNG_CHANNELS = 3


font = cv2.FONT_HERSHEY_COMPLEX_SMALL
fontScale = 0.45
fontColor = (255, 255, 255)
lineType = 1

png = np.zeros((PNG_HEIGHT, PNG_WIDTH, PNG_CHANNELS))
print("Allocated array of shape {}".format(png.shape))

# find all that are unprocessed in detections
cursor = detections.find({
    'processed': 1
})

mem_detected = {}
curr_img = 0
for detection in cursor.limit(COLS_PER_ROW*ROWS_PER_IMG):

    t_row = int(curr_img / COLS_PER_ROW)
    t_col = int(curr_img % COLS_PER_ROW)
    chip = pickle.loads(detection['chip'])

    png_y = t_row*CHIP_HEIGHT
    png_x = t_col*CHIP_WIDTH

    # fit face to grid
    png[png_y:png_y+CHIP_HEIGHT, png_x:png_x+CHIP_WIDTH] = chip

    # overla name in cell
    bottomLeftCornerOfText = (png_x, png_y+10)

    cv2.putText(png, mem_persons[detection['person_id']],
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
                8
                )

    curr_img = curr_img + 1


time_end = time.time()
cv2.imwrite("{}-chips.png".format(time.time()), png)

print("\nProcessing completed... took [{}] s\n".format(time_end - time_start))
