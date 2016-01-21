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
        v_points.append(sphere(pos=points[i],radius= 3e-4 * (masses[i])**0.3, color= color.white, make_trail=True, material=materials.shiny, retain=400))
        print (points[i])

while true :#updating position
    for data in datas:
            rate(50)
            points=data.split('|')
            points=[tuple(map(float,p.split(';'))) for p in points]
            newcenter = [ 0,0,0 ]
            for i in range(len(v_points)):
                for j in [0,1,2]:
                    newcenter[j] += points[i][j]
                v_points[i].pos= points[i]
            for j in [0,1,2]:
                newcenter[j] /= len(v_points)
            
            scene.center = tuple( newcenter )

