import numpy as np
import cv2
from decimal import *
import pickle
getcontext().prec = 60

# I needed a function like this to help me trace the progress of the encoding and decoding
# so I search and get it from an answer on stackoverflow. It's just a utility function
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 0, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

validImageName = False
while not validImageName:
    try:
        imageName = input("⚙️ Enter image file name : ")
        img = cv2.imread(imageName, 0)
        if(type(img) == type(None)):
            raise Exception("Can't find this image")
        validImageName = True 
    except:
        print("❌ Can't find this file, Please enter a valid file name")

validBlockSize = False
while not validBlockSize:
    try:
        blockSize = input("⚙️ Enter block size : ")
        blockSize = int(blockSize)
        validBlockSize = True 
    except:
        print("❌ This blocksize isn't valid, Please enter an intger")

imgOriginal = img
img = img.flatten();

probabilities = {}
totalPixels = len(img)

zerosAtEnd = blockSize - ( blockSize if totalPixels%blockSize == 0 else totalPixels%blockSize)
img = np.append(img, np.zeros(zerosAtEnd))

for colorDegree in img:
    if colorDegree in probabilities:
        probabilities[colorDegree] += 1
    else:
        probabilities[colorDegree] = 1

for colorDegree in probabilities:
  probabilities[colorDegree] = Decimal( probabilities[colorDegree] ) / Decimal( totalPixels )


# a function that takes a symbol and the probability dictionary and returns the cumulative probability
def getSumProb(symbol, probDic):
    sum = 0
    probSymbol = 0

    for s in probDic:
        if(s == symbol):
            probSymbol = probDic[s]
            break
        else :
            sum += probDic[s]
    
    sumWithSymbol = sum + probSymbol

    return [sum, sumWithSymbol ]

# a dictionary with the cumulative probabilty of each symbol
cumulativeProbDic = {}

for symbol in probabilities:
    cumulativeProbDic[symbol] = getSumProb(symbol, probabilities)

imgReshaped = np.reshape(img, (-1,blockSize))

def arithmeticCode(sentence, prevLowerLimit, prevUpperLimit, probabilitiesDic, cumulativeProbDic):    
    currentLowerLimit = prevLowerLimit + ( prevUpperLimit - prevLowerLimit )*cumulativeProbDic[ sentence[0] ][0]
    currentUpperLimit = prevLowerLimit + ( prevUpperLimit - prevLowerLimit )*cumulativeProbDic[ sentence[0] ][1]
    
    if( len(sentence) == 1 ):
        return ( currentLowerLimit + currentUpperLimit ) / Decimal(2)
    else:
        return arithmeticCode(sentence[1:], currentLowerLimit, currentUpperLimit, probabilitiesDic, cumulativeProbDic)
    
tags=[]
for x in imgReshaped:
    tags.append( arithmeticCode(x, Decimal(0), Decimal(1), probabilities, cumulativeProbDic) )


print('⏳Generating files to be used by decoder ...')
# Generating probabilities dictionary file
file = open("probabilities.pkl","wb")
pickle.dump(probabilities,file)
file.close()


# output the image info file
imgInfo = np.array( [imgOriginal.shape] + [blockSize] + [zerosAtEnd] + tags  )
np.save( 'imageInfo.npy',  imgInfo)
