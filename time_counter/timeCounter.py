import csv
import time
import tkinter
import cv2
import matplotlib.pyplot
import numpy
import pandas
from PIL import Image,ImageTk
from face_racgnization import re

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

def tkVideo():
    ref,frame = cap.read()
    frame = cv2.flip(frame,1)
    catch_frame = re.catch_face(frame)
    cvImage = cv2.cvtColor(catch_frame,cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(cvImage)
    pilImage = pilImage.resize((image_width,image_height),Image.ANTIALIAS)
    tkVideo = ImageTk.PhotoImage(image=pilImage)
    return tkVideo

def start():
    global time_start
    time_start = time.time()
    btnStart['state'] = tkinter.DISABLED
    btnEnd['state'] = tkinter.NORMAL


def end():
    time_end = time.time()
    name = e1.get()
    calculate(name,time_end)
    btnStart['state'] = tkinter.NORMAL
    btnEnd['state'] = tkinter.DISABLED



def calculate(name,time_end):
    total = time_end-time_start
    header = ['name','cost']
    flag = 0
    dic = {'name':name,'cost':total}

    with open('test.csv',newline='',encoding='utf-8') as f:
        data = [row for row in csv.DictReader(f)]
        for item in data:
            if item['name'] == name:
                item['cost'] = round(float(item['cost'])+total,2)
                print(item['cost'])
                print(float(item['cost']))
                print(total)
                flag = 1
        if flag == 0:
            data.append(dic)
        f.close()

    with open('test.csv','w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
        writer.writeheader()
        writer.writerows(data)


def show():
    data = pandas.read_csv('test.csv')
    data = numpy.array(data)

    print(data)
    data[:,1] = data[:,1]/3600
    matplotlib.pyplot.subplot(1,2,1)
    matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
    matplotlib.pyplot.pie(data[:,1],labels=data[:,0],autopct = '%.2f%%')
    matplotlib.pyplot.subplot(1, 2, 2)
    for i in range(data.shape[0]):
        matplotlib.pyplot.bar(data[i,0],data[i,1])
    matplotlib.pyplot.xlabel("event`name")
    matplotlib.pyplot.ylabel("time cost")
    matplotlib.pyplot.title("calculate cost time bar map")
    matplotlib.pyplot.show()



base = tkinter.Tk()
base.wm_title('counter')
base.geometry('900x600')
btnStart = tkinter.Button(base,text='开始',command=start)
btnEnd = tkinter.Button(base, text='结束', command=end,state=tkinter.DISABLED)
btnShow = tkinter.Button(base, text='展示饼图', command=show)


l = tkinter.Label(base,text='Event')
l.grid(row=2,column=1)
e1 = tkinter.Entry(base)
e1.grid(row=2,column=2)

btnStart.grid(row=3,column=2)
btnEnd.grid(row=4,column=2)
btnShow.grid(row=5,column=2)
image_width = 600
image_height = 500
canvas = tkinter.Canvas(base,bg='white',width=image_width,height=image_height)
canvas.grid(row=2)

while True:
    pic = tkVideo()
    canvas.create_image(0,0,anchor='nw',image = pic)
    base.update()
    base.after(1)

base.mainloop()

