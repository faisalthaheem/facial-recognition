import time
from bson.binary import Binary
import pickle
from pymongo import MongoClient
import numpy as np
import pprint
import tqdm
import logging
import argparse
import names
import uuid

logging.basicConfig(level=0)

ap = argparse.ArgumentParser()
ap.add_argument("-du", "--mongo.url", default="mongodb://localhost:27017",
                help="db to store our detections to.")
args = vars(ap.parse_args())
LOG = logging.getLogger(__name__)

mongoClient = MongoClient(args['mongo.url'])
mongodb = mongoClient.facials
detections = mongodb.detections
persons = mongodb.persons

count_new = 0
count_update = 0

time_start = time.time()

mem_persons = {}
# find all persons
cursor = persons.find({})
for doc in cursor:
    mem_persons[doc['_id']] = pickle.loads(doc['signature'])
print("loaded [{}] persons".format(len(mem_persons)))

# find all that are unprocessed in detections
cursor = detections.find({
    'processed': 0
})

mem_detected = {}
for detection in cursor:
    mem_detected[detection['_id']] = pickle.loads(detection['signature'])
print("loaded [{}] unprocessed events".format(len(mem_detected)))


def findMatch(lstOfKnownPersons, detectionSignature):

    for kp, vp in lstOfKnownPersons.items():
        dist = np.linalg.norm(vp - detectionSignature)
        if dist <= 0.4:
            return True, kp

    return False, None


new_persons = {}
for kd, vd in mem_detected.items():
    matched = False

    # lookup persons  from db
    matched, kp = findMatch(mem_persons, vd)

    # look up persons identified in current run
    if not matched:
        matched, kp = findMatch(new_persons, vd)

    # no match found, insert into new_persons
    if not matched:
        kp = uuid.uuid4()
        new_persons[kp] = vd
    else:
        count_update = count_update + 1

    detections.update_one(
        {'_id': kd}, {"$set": {"person_id": kp, "processed": 1}})


for k, v in new_persons.items():

    persons.insert_one({
        "_id": k,
        "signature": Binary(pickle.dumps(v, protocol=2), subtype=128),
        "full_name": names.get_full_name(),
        "created": time.time()
    })
    count_new = count_new + 1

time_end = time.time()

print("\nProcessing completed... took [{}] s\n".format(time_end - time_start))
print("Inserted [{}]".format(count_new))
print("Updated [{}]".format(count_update))
