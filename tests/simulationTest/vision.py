import numpy as np
from visual import*

def main():
    #get datas
    from sys import argv
    filename = "test.txt"
    framerate = 50
    if len(argv) > 1:
        filename = argv[1]
    if len(argv) > 2:
        framerate = argv[2]
    with open(filename) as f:
        datas = f.read()

    #Pharse Datas
    datas= datas.split('\n')
    masses=datas.pop(0).split(' ')
    masses=[(float(mass))**0.3 for mass in masses]
    datas.pop()

    #initialisierung
    points=datas.pop(0).split('|')
    points= [tuple(map(float,p.split(';'))) for p in points]
    v_points=[]
    for i in range(len(points)):
        v_points.append( sphere(pos=points[i],radius= 4 * masses[i], color = color.white,  make_trail=True, material=materials.diffuse, retain=400))

    #update Positions
    while True :
        for data in datas:
            rate(framerate)
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

main()
