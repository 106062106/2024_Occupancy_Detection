"""
    This modules provides methods to manage and manipulate mp4 files.
"""

import os
import traceback

import cv2 as cv

import ultralytics as UL

def list_path(dirPth: str):
    """
        Lists all mp4 file paths under given directory.

        Keyword Arguments:
        :dirPth (str) -- the lookup directory path
    """
    try:
        pthLst = []
        for curDir, _, filLst in os.walk(dirPth):
            for file in filLst:
                if file.endswith(".mp4"):
                    pth = os.path.join(curDir, file)
                    pthLst.append([pth])

        return pthLst
    except Exception as exc:
        traceback.print_exception(exc)
        return None

def change_fps(src: str, fps: int, dst: str):
    """
        Change mp4 FPS (Frame Per Second)
        by the provided source, fps and destination.

        Keyword Arguments
        :src (str) -- the source video path
        :fps (int) -- assigning fps
        :dst (str) -- the destination video path
    """
    try:
        cap = cv.VideoCapture(src)
        vid = cv.VideoWriter(
            dst,
            int(cap.get(cv.CAP_PROP_FOURCC)),
            fps,
            (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        )

        while cap.isOpened():
            vld, frm = cap.read()
            if not vld:
                break
            vid.write(frm)

        cap.release()
        vid.release()
        cv.destroyAllWindows()
    except Exception as exc:
        traceback.print_exception(exc)

def count_people_per_sec(src: str, iou = 0.7):
    """
        Counts the number of people in the video per second,
        by using YOLOv8 provided by ultralytics.

        Keyword Arguments
        :src (str) -- the source mp4 file path
        :iou (float) -- intersection over union threshold for Non-Maximum Suppression (NMS)
    """
    try:
        vid = cv.VideoCapture(src)
        fps = int(vid.get(cv.CAP_PROP_FPS))

        mdl = UL.YOLO("yolov8n.pt")

        fpsRes = 1
        cntLst = []
        while vid.isOpened():
            vld, frm = vid.read()
            if not vld:
                break
            
            trkLst = mdl.track(
                frm,
                persist = True,
                show = False,
                classes = 0,
                verbose = False,
                iou = iou
            )

            if fpsRes > 1:
                fpsRes -= 1
                continue
            else:
                fpsRes = fps
                cntLst.append([len(trkLst[0].boxes)])
        
        return cntLst
    except Exception as exc:
        traceback.print_exception(exc)
        return None
