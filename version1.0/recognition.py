import data_load
import cv2 as cv
import PCA
import svm

# 启动摄像头，实时进行人脸识别
# 参数 ： 文件路径名称

def FaceRecognize(data):
    # 加载训练数据
    X, y, names = data_load.LoadData(data)

    model = svm.svc(data)
    print(model)
    face_cascade = cv.CascadeClassifier('C:\Dev\Python\\venvs\work\Lib\site-packages\cv2\data\\haarcascade_frontalface_default.xml')

    # 打开摄像头
    camera = cv.VideoCapture(0)
    cv.namedWindow('Face')

    while (True):
        # 读取一帧图像
        ret, frame = camera.read()
        # 判断图片读取成功
        if ret:
            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # 人脸检测

            faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
            for (x, y, w, h) in faces:
                # 在原图像上绘制矩形
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray_img[y:y + h, x:x + w]

                # 宽92 高112
                roi_gray = cv.resize(roi_gray, (92, 112), interpolation=cv.INTER_LINEAR)
                roi_gray = PCA.PCA_Data(roi_gray)

                roi_gray = roi_gray.ravel()
                roi_gray = roi_gray.reshape(1,-1)
                print(roi_gray)
                label = model.predict(roi_gray)
                print(label)
                print(names)
                # print('Label:%s,confidence:%.2f' % (params[0], params[1]))
                cv.putText(frame, names[label[0]], (x, y - 20), cv.FONT_HERSHEY_SIMPLEX, 1, 255, 2)


            cv.imshow('Face', frame)
            # 如果按下q键则退出
            if cv.waitKey(100) & 0xff == ord('q'):
                break
    camera.release()
    cv.destroyAllWindows()