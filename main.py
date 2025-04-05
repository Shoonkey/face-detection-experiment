import cv2 as cv
import sys


def main():

    if len(sys.argv) < 2:
        print("Please pass the path to the video file as the first argument!")
        return

    video_filename = sys.argv[1]

    # initiate video processor and get resolution data
    video_capture = cv.VideoCapture(video_filename)
    video_width = video_capture.get(cv.CAP_PROP_FRAME_WIDTH)
    video_height = video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)

    # initiate face classifier model
    CLASSIFIERS_PATH = cv.data.haarcascades
    FRONTAL_FACE = "haarcascade_frontalface_alt.xml"
    face_classifier = cv.CascadeClassifier(CLASSIFIERS_PATH + FRONTAL_FACE)

    VIDEO_WINDOW_TITLE = "Face detection using Haar cascades (Press Q to close)"

    while True:
        result, video_frame = video_capture.read()

        if not result:
            break

        # downscale it (halves the resolution) to make it faster and also have the full HD video easily fit the screen
        downscale_division_factor = 3
        
        new_video_width = int(video_width // downscale_division_factor)
        new_video_height = int(video_height // downscale_division_factor)
        new_resolution = (new_video_width, new_video_height)

        halve_resolution_matrix = cv.getRotationMatrix2D((0, 0), 0, 1 / downscale_division_factor)
        video_frame = cv.warpAffine(video_frame, halve_resolution_matrix, new_resolution)

        # preprocess frame into grayscale to use less computational power, since the algorithm is optimized for that
        grayscale_frame = cv.cvtColor(video_frame, cv.COLOR_BGR2GRAY)

        # run detection and get faces' bounding boxes
        face_rectangles = face_classifier.detectMultiScale(
            grayscale_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv.CASCADE_SCALE_IMAGE,
        )

        # draw rectangles around the faces
        PINK = (255, 156, 210)
        for rect in face_rectangles:
            cv.rectangle(video_frame, rect, color=PINK, thickness=2)

        cv.imshow(VIDEO_WINDOW_TITLE, video_frame)

        # if a key was pressed and it's Q, then finish the loop
        pressed_key = cv.waitKey(20)
        if pressed_key != -1 and chr(pressed_key).upper() == "Q":
            break
    
    video_capture.release()
    cv.destroyWindow(VIDEO_WINDOW_TITLE)


if __name__ == "__main__":
    main()
