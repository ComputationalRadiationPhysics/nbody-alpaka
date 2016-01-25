import numpy as np
from visual import*

def main(filename, framerate):
    #get datas
    with open(filename) as f:
        datas = f.read()

    #Parse Datas
    lines = datas.split('\n')
    masses=[(float(mass)) for mass in lines[0].split(' ')]

    steps = []
    for line in lines[1:]:
        if len(line.strip()) < 3:
            continue
        bodies = []
        for bodyinfo in line.split('|'):
            point = tuple(map(float,bodyinfo.split(';')))
            bodies.append(point)
        steps.append(bodies)

    colors = [
        color.red,
        color.yellow,
        color.green,
        color.orange,
        color.white,
        color.blue,
        color.magenta
    ]

    #Run steps
    while True:
        v_points=[]
        for i,position in enumerate(steps[0]):
            objectcolor = colors[ i % len(colors) ]
            object = sphere(pos=position,radius= 4 * masses[i]**0.3, color = objectcolor, material=materials.diffuse, make_trail = True)
            v_points.append( object )

        for step in steps:
            if len(step) != len(v_points):
                print "Invalid step found"
                break
            rate(framerate)
            newcenter = [ 0,0,0 ]
            for i, position in enumerate(step):
                for j in range(3):
                    newcenter[j] += position[j]
                v_points[i].pos = position
            for j in range(3):
                newcenter[j] /= len(step)

            scene.center = tuple( newcenter )

        while len(v_points) > 0:
            v_points[0].visible = False
            v_points[0].trail_object.visible = False
            del v_points[0]

if __name__ == '__main__':
    from sys import argv
    filename = "test.txt"
    framerate = 50
    if len(argv) > 1:
        filename = argv[1]
    if len(argv) > 2:
        framerate = argv[2]

    main(filename, framerate)
