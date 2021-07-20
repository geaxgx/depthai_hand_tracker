#
# Filtering
#
from math import pi
import numpy as np
import time

class LandmarksSmoothingFilter: 
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmarks_smoothing_calculator.cc
    
    frequency, min_cutoff, beta, derivate_cutoff: 
                See class OneEuroFilter description.
    min_allowed_object_scale:
                If calculated object scale is less than given value smoothing will be
                disabled and landmarks will be returned as is. Default=1e-6
    disable_value_scaling:
                Disable value scaling based on object size and use `1.0` instead.
                If not disabled, value scale is calculated as inverse value of object
                size. Object size is calculated as maximum side of rectangular bounding
                box of the object in XY plane. Default=False
    '''
    def __init__(self,
                frequency=30,
                min_cutoff=1,
                beta=0,
                derivate_cutoff=1,
                min_allowed_object_scale=1e-6,
                disable_value_scaling=False
                ):
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivate_cutoff = derivate_cutoff
        self.min_allowed_object_scale = min_allowed_object_scale
        self.disable_value_scaling = disable_value_scaling
        self.init = True

    @staticmethod
    def get_object_scale(landmarks):
        # Estimate object scale to use its inverse value as velocity scale for
        # RelativeVelocityFilter. If value will be too small (less than
        # `options_.min_allowed_object_scale`) smoothing will be disabled and
        # landmarks will be returned as is.
        # Object scale is calculated as average between bounding box width and height
        # with sides parallel to axis.
        min_xy = np.min(landmarks[:,:2], axis=0)
        max_xy = np.max(landmarks[:,:2], axis=0)
        return np.mean(max_xy - min_xy)

    def apply(self, landmarks, timestamp=None, object_scale=0):
        # object_scale: in practice, we use the size of the rotated rectangle region.rect_w_a=region.rect_h_a

        if timestamp is None: timestamp = time.perf_counter()
        # Initialize filters 
        if self.init:
            self.filters = OneEuroFilter(self.frequency, self.min_cutoff, self.beta, self.derivate_cutoff)
            self.init = False

        # Get value scale as inverse value of the object scale.
        # If value is too small smoothing will be disabled and landmarks will be
        # returned as is.  
        if self.disable_value_scaling:
            value_scale = 1
        else:
            object_scale = object_scale if object_scale else self.get_object_scale(landmarks) 
            if object_scale < self.min_allowed_object_scale:
                return landmarks
            value_scale = 1 / object_scale

        return self.filters.apply(landmarks, value_scale, timestamp)

    def get_alpha(self, cutoff):
        '''
        te = 1.0 / self.frequency
        tau = 1.0 / (2 * Math.PI * cutoff)
        result = 1 / (1.0 + (tau / te))
        '''
        return 1.0 / (1.0 + (self.frequency / (2 * pi * cutoff)))

    def reset(self):
        self.init = True

class OneEuroFilter: 
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/one_euro_filter.cc
    Paper: https://cristal.univ-lille.fr/~casiez/1euro/

    frequency:  
                Frequency of incoming frames defined in seconds. Used
                only if can't be calculated from provided events (e.g.
                on the very first frame). Default=30
    min_cutoff:  
                Minimum cutoff frequency. Start by tuning this parameter while
                keeping `beta=0` to reduce jittering to the desired level. 1Hz
                (the default value) is a a good starting point.
    beta:       
                Cutoff slope. After `min_cutoff` is configured, start
                increasing `beta` value to reduce the lag introduced by the
                `min_cutoff`. Find the desired balance between jittering and lag. Default=0
    derivate_cutoff: 
                Cutoff frequency for derivate. It is set to 1Hz in the
                original algorithm, but can be turned to further smooth the
                speed (i.e. derivate) on the object. Default=1
    '''
    def __init__(self,
                frequency=30,
                min_cutoff=1,
                beta=0,
                derivate_cutoff=1,
                ):
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivate_cutoff = derivate_cutoff
        self.x = LowPassFilter(self.get_alpha(min_cutoff))
        self.dx = LowPassFilter(self.get_alpha(derivate_cutoff))
        self.last_timestamp = 0

    def get_alpha(self, cutoff):
        '''
        te = 1.0 / self.frequency
        tau = 1.0 / (2 * Math.PI * cutoff)
        result = 1 / (1.0 + (tau / te))
        '''
        return 1.0 / (1.0 + (self.frequency / (2 * pi * cutoff)))

    def apply(self, value, value_scale, timestamp):
        '''
        Applies filter to the value.
        timestamp in s associated with the value (for instance,
        timestamp of the frame where you got value from).
        '''
        if self.last_timestamp >= timestamp:
            # Results are unpreditable in this case, so nothing to do but return same value.
            return value

        # Update the sampling frequency based on timestamps.
        if self.last_timestamp != 0 and timestamp != 0:
            self.frequency = 1 / (timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

        # Estimate the current variation per second.
        if self.x.has_last_raw_value():
            dvalue = (value - self.x.last_raw_value()) * value_scale * self.frequency
        else:
            dvalue = 0
        edvalue = self.dx.apply_with_alpha(dvalue, self.get_alpha(self.derivate_cutoff))

        # Use it to update the cutoff frequency
        cutoff = self.min_cutoff + self.beta * np.abs(edvalue)

        # filter the given value.
        return self.x.apply_with_alpha(value, self.get_alpha(cutoff))
        
class LowPassFilter:
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/low_pass_filter.cc
    Note that 'value' can be a numpy array
    '''
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.initialized = False

    def apply(self, value):
        if self.initialized:
            # Regular lowpass filter.
            # result = alpha * value + (1 - alpha) * stored_value;
            result = self.alpha * value + (1 - self.alpha) * self.stored_value
        else:
            result = value
            self.initialized = True
        self.raw_value = value
        self.stored_value = result
        return result

    def apply_with_alpha(self, value, alpha):
        self.alpha = alpha
        return self.apply(value)

    def has_last_raw_value(self):
        return self.initialized

    def last_raw_value(self):
        return self.raw_value

    def last_value(self):
        return self.stored_value

