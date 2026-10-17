import cv2

import cam

ctis = cam.load_cti_envvar()
h, ia = cam.setup_cam(ctis)

try:
    while True:
        cv2.imshow('LiveView', cam.get_harvester_image(ia))
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    cam.teardown_cam(h, ia)
    cv2.destroyAllWindows()
