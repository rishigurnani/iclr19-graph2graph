#import
import sys
sys.path.append('/home/rgur/py_scripts/')
sys.path.append('/home/rishi/py_scripts/')
import rishi_utils as ru
#perform tasks
ys = []
xs = []
lines = []
for line in sys.stdin:
    if line not in lines: #delete duplicate source:target pairs
        lines.append(line)
        x,y = line.split()
        if y != None:
            poly_ys = ru.polymerize(y)
            if poly_ys != None:
                for p in poly_ys:
                    print x, p
        