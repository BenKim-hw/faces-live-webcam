import utils
import yolo_face

if __name__=='__main__':
    argv = utils.parse()
    yolo_face.run(argv)
    exit(0)