import cv2, numpy as np


class SimpleGridworld(object):

    def __init__(self):
        self.position = np.array([0,0], dtype=np.int32)


    def step(self, a):
        action_mapping = {0: (-1,0), 1: (1,0), 2: (0, -1), 3: (0, 1)}
        delta = action_mapping[a]
        new_position = self.position + np.array(delta)
        new_position = np.clip(new_position, 0, 11)
        position = new_position
        return self.get_observation(position)


    def get_observation(self, position):
        canvas = np.zeros([12, 12, 1], np.float32)
        y, x = position[0], position[1]
        canvas[y:y+2, x:x+2, :] = 1
        return canvas

    def get_position(self):
        return np.copy(self.position)

    def set_position(self, new_position):
        self.position = np.copy(new_position)


