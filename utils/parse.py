import argparse

def parse():
    parser = argparse.ArgumentParser(description='FACE-WEBCAM')
    parser.add_argument('-t', '--face_thr', type=float, default=0.5)
    parser.add_argument('-u', '--close_thr', type=float, default=1.0)
    parser.add_argument('-m', '--min_size', type=int, default=80)
    parser.add_argument('-f', '--frames_label', type=int, default=5)
    
    argv = parser.parse_args()

    return argv