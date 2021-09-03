#!/usr/bin/env python3


from HandTrackerRenderer import HandTrackerRenderer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str,
                    help="Path to a blob file for palm detection model")
parser_tracker.add_argument('--no_lm', action="store_true", 
                    help="Only the palm detection model is run (no hand landmark model)")
parser_tracker.add_argument("--lm_model", type=str,
                    help="Path to a blob file for landmark model")
parser_tracker.add_argument('-s', '--solo', action="store_true", 
                    help="Detect one hand max. Default in solo mode.")                    
parser_tracker.add_argument('-xyz', "--xyz", action="store_true", 
                    help="Enable spatial location measure of palm centers")
parser_tracker.add_argument('-g', '--gesture', action="store_true", 
                    help="Enable gesture recognition")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument("-r", "--resolution", choices=['full', 'ultra'], default='full',
                    help="Sensor resolution: 'full' (1920x1080) or 'ultra' (3840x2160) (default=%(default)s)")
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels") 
parser_tracker.add_argument("-bpf", "--body_pre_focusing", choices=['right', 'left', 'group', 'higher'],
                    help="Enable Body Pre Focusing")      
parser_tracker.add_argument('-ah', '--all_hands', action="store_true", 
                    help="In Body Pre Focusing mode, consider all hands (not only the hands up)")                                     
parser_tracker.add_argument('-t', '--trace', action="store_true", 
                    help="Print some debug messages")                
parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-o', '--output', 
                    help="Path to output video file")
args = parser.parse_args()

if args.edge:
    from HandTrackerEdge import HandTracker
else:
    from HandTracker import HandTracker

dargs = vars(args)
tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

tracker = HandTracker(
        input_src=args.input, 
        use_lm= not args.no_lm, 
        use_gesture=args.gesture,
        xyz=args.xyz,
        solo=args.solo,
        crop=args.crop,
        resolution=args.resolution,
        body_pre_focusing=args.body_pre_focusing,
        hands_up_only=not args.all_hands,
        stats=True,
        trace=args.trace,
        **tracker_args
        )

renderer = HandTrackerRenderer(
        tracker=tracker,
        output=args.output)

while True:
    # Run hand tracker on next frame
    # 'bag' is information common to the frame and to the hands 
    # (like body keypoints in Body Pre Focusing mode)
    frame, hands, bag = tracker.next_frame()
    if frame is None: break
    # Draw hands
    frame = renderer.draw(frame, hands, bag)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
tracker.exit()
