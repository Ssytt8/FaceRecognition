import cv2 as cv
import os

# 生成自己的人脸数据 保存到指定目录中
# 参数 ： 数据保存的位置
def generator(data):
    name = input('Input Name: ')

    # 拼接路径
    path = os.path.join(data, name)
    # 如果路径存在则删除
    # if os.path.isdir(path):
    #     shutil.rmtree(path) #递归删除文件夹

    # 读取到该文件下 已经存在的最大index
    index = 0
    if os.path.exists(path):
        for subDirname in os.listdir(data):
            subjectPath = os.path.join(data, subDirname)
            if os.path.isdir(subjectPath):
                # 每一个文件夹下存放着一个人的照片
                for fileName in os.listdir(subjectPath):
                    index += 1

    # 如果没有文件夹  创建文件
    if not os.path.exists(path):
        os.mkdir(path)

    # 创建一个级联分类器，加载一个xml分类器文件，它既可以是Harr特征也可以是LBP特征的分类器
    face_cascade = cv.CascadeClassifier('C:\Dev\Python\\venvs\work\Lib\site-packages\cv2\data\\haarcascade_frontalface_default.xml')

    # 打开摄像头
    camera = cv.VideoCapture(0)
    cv.namedWindow('Face')


    # 计数
    count = 0
    while True:
        # 读取一帧图像
        ret, frame = camera.read()

        # 判断图片是否读取成功
        if ret:
            gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            #人脸检测
            faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
            for (x,y,w,h) in faces:
                print(x,y,w,h)

                # 在原图像上绘制矩形
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                # 调整图像大小 和ORL人脸库图像一样大小
                f = cv.resize(frame[y:y+h,x:x+w],(92,112))

                # 保存人脸
                cv.imwrite('%s/%s.jpg'%(path,str(count+index)),f)
                count += 1
            cv.imshow('Face', frame)

            #如果按下q键则退出
            if cv.waitKey(100) & 0xff == ord('q') or count == 20:
                break
    camera.release()
    cv.destroyAllWindows()


