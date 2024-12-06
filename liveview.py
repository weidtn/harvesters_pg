import cv2
import numpy as np
import cam

ctis = cam.find_common_cti_files()
h, ia = cam.setup_cam(ctis[0])

try:
    while True:
        with ia.fetch() as buffer:
            component = buffer.payload.components[0]
            img = component.dataimg = buffer.payload.components[0].data
            img = np.reshape(img, (buffer.payload.components[0].height, buffer.payload.components[0].width))
            cv2.imshow('LiveView', img)
            if cv2.waitKey(1) == ord('q'):
                break
finally:
    cam.disconnect_cam(h, ia)
    cv2.destroyAllWindows()
