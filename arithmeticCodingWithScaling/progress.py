import time

def printProgress(current, range):
    print (' ⌛️Loading :' , round( (current/range)*100, 2) , '%' , end="\r")

for i in range(0, 20):
    printProgress(i, 20)
    time.sleep(1)
    