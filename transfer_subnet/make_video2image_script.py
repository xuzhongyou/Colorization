import os
import cv2

# huxiaoke 自己做的数据集
def video2image(in_path,out_path,img_shape=(640,400)):

    for video_path in os.listdir(in_path):
        if video_path[:2] == '._':
            continue
        video = cv2.VideoCapture(os.path.join(in_path, video_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        flag = video.isOpened()
        print('Read the video {} is successful ? and its fps is {}  :{}'.format(video_path,fps,flag))
        index = 1
        dir_path = os.path.join(out_path,video_path[:-5])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        while (flag):
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.resize(frame, img_shape)
            if index < 10:
                frame_path = 'frame_000{}.png'.format(index)
            else :
                frame_path = 'frame_00{}.png'.format(index)
            image_path = os.path.join(dir_path,frame_path)
            print('It is saving frame {} now and its index is {}!!'.format(image_path,index))
            cv2.imwrite(image_path,frame)
            index +=1


# /video/data/FBMS/Trainingset/bear01
def batch_image_resize(in_path,out_path,image_shape=(640,400)):
    for img_dir in os.listdir(in_path):
        img_dir_in_path = os.path.join(in_path,img_dir)
        img_dir_out_path = os.path.join(out_path,img_dir)
        print('It is now processing img_dir_in_path:  ',img_dir_in_path)
        print('It is saving img_dir_out_path:  ',img_dir_out_path)
        for item in os.listdir(img_dir_in_path):
            item_in_path = os.path.join(img_dir_in_path,item)
            print('It is reading image item_in_path:  ',item_in_path)
            image = cv2.imread(item_in_path)
            image = cv2.resize(image,image_shape)
            if not os.path.exists(img_dir_out_path):
                os.makedirs(img_dir_out_path)
            item_out_path = os.path.join(img_dir_out_path,item)
            cv2.imwrite(item_out_path,image)



if __name__ == "__main__":
    in_path = '/home/xzy/video/data/DAVIS-test-raw/JPEGImages/Full-Resolution'

    out_path = '/home/xzy/video/data/DAVIS-test'
    batch_image_resize(in_path,out_path)