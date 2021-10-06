#!/usr/bin/env python3

import numpy as np
import depthai as dai
import cv2
from pathlib import Path
import time

SCRIPT_DIR = Path(__file__).resolve().parent
LANDMARK_MODEL = str(SCRIPT_DIR / "models/hand_landmark_sh4.blob")

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1)#.flatten()

class HandTracker:
    def __init__(self, input_src=None,
                lm_model=LANDMARK_MODEL,
                ):

        self.lm_model = lm_model
        print(f"Landmark blob       : {self.lm_model}")
        self.device = dai.Device()
        self.img = cv2.imread(input_src)
        
        # Define and start pipeline
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
        self.q_lm_in = self.device.getInputQueue(name="lm_in")


    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)
        
         # Define hand landmark model
        print("Creating Hand Landmark Neural Network...")          
        lm_nn = pipeline.createNeuralNetwork()
        lm_nn.setBlobPath(self.lm_model)
        lm_nn.setNumInferenceThreads(2)
        # Hand landmark input
        self.lm_input_length = 224
        lm_in = pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_in.out.link(lm_nn.input)
        # Hand landmark output
        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_nn.out.link(lm_out.input)
            
        print("Pipeline created.")
        return pipeline        

    def next_frame(self):
        nb_hands = 2
        for i in range(nb_hands):
            nn_data = dai.NNData()   
            nn_data.setLayer("input_1", to_planar(self.img, (self.lm_input_length, self.lm_input_length)))
            self.q_lm_in.send(nn_data)
            # time.sleep(0.002)

        for i in range(nb_hands):
            inference = self.q_lm_out.get()
            lm_score = inference.getLayerFp16("Identity_1")[0]  
            print(f"lm score hand {i}: {lm_score}")
    

tracker = HandTracker(input_src='hand0.jpg')

while True:
    tracker.next_frame()