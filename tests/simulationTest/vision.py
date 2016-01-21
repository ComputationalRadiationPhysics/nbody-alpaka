import numpy as np
from visual import*

with open("test.txt") as f:
    datas = f.read()
datas= datas.split('\n')
masses=datas.pop(0).split(' ')
masses=[1000000000000000 for mass in masses]
datas.pop()
#initzialisierung
points=datas.pop(0).split('|')
points= [tuple(map(float,p.split(';'))) for p in points]
v_points=[]
for i in range(len(points)):
        v_points.append(sphere(pos=points[i],radius= masses[i], color= color.red))
        print (points[i])

while true :#updating position
    for data in datas:
            rate(10)
            points=data.split('|')
            points=[tuple(map(float,p.split(';'))) for p in points]
            for i in range(len(v_points)):
                    v_points[i].pos= points[i]

