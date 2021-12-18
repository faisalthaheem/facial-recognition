import sys
import os
import shutil as sh
import logging
import numpy as np


class FaceDb:
    def __init__(self, logger, path_db):
        self.logger = logger
        self.path_db = path_db
        self.db = {}

    def load(self):

        if not os.path.exists(self.path_db) or not os.path.isdir(self.path_db):
            self.logger.error(
                "%s is not a directory or does not exist." % self.path_db)
            return False

        self.logger.info("Processing files in " + self.path_db)
        for root, dirs, files in os.walk(self.path_db):

            totalFiles = len(files)
            for i in range(0, totalFiles):
                fileName = files[i]
                srcPath = os.path.join(root, fileName)
                self.db[fileName] = np.loadtxt(srcPath)

            self.logger.info(
                "[{}] files to loaded from [{}]".format(totalFiles, root))

    def match(self, sig_to_match, thresh=0.4):

        for key in self.db:
            # print(self.db[key])
            dist = np.linalg.norm(self.db[key] - sig_to_match)
            if dist <= thresh:
                return key, dist

        return None, None
