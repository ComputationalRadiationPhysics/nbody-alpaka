import numpy as np
from visual import*

def main():
    #get datas
    from sys import argv
    filename = "test.txt"
    framerate =10
    if len(argv) > 1:
        filename = argv[1]
    if len(argv) > 2:
        framerate = argv[2]
    with open(filename) as f:
        datas = f.read()
    
    #Pharse Datas
    datas= datas.split('\n')
    masses=datas.pop(0).split(' ')
    masses=[(int(mass))**0.3 for mass in masses]
    datas.pop()
    
    #initzialisierung
    points=datas.pop(0).split('|')
    points= [tuple(map(float,p.split(';'))) for p in points]
    v_points=[]
    for i in range(len(points)):
            v_points.append(sphere(pos=points[i],radius= masses[i], color= color.red))

    #upadate Positions
    while true :
        for data in datas:
                rate(framerate)
                points=data.split('|')
                points=[tuple(map(float,p.split(';'))) for p in points]
                for i in range(len(v_points)):
                        v_points[i].pos= points[i]

