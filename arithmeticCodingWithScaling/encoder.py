import numpy as np
import cv2
import pickle

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

# read the image and convert it to grayscale

imgOriginal = img
img = img.flatten()
blockSize = 16

# probabilities dictionary
probabilities = {}
totalPixels = len(img)

# looping over each pixel of the image to get the frequency of each color
for colorDegree in img:
    if colorDegree in probabilities:
        probabilities[colorDegree] += 1
    else:
        probabilities[colorDegree] = 1

# looping over color degrees to get the proabability of each color degree
for colorDegree in probabilities:
    probabilities[colorDegree] = probabilities[colorDegree] / totalPixels

# the number of zeros at end for padding
zerosAtEnd = blockSize - (blockSize if totalPixels % blockSize == 0 else totalPixels % blockSize)
img = np.append(img, np.zeros(zerosAtEnd))

# a function that takes a symbol and the probability dictionary and returns the cumulative probability
def getSumProb(symbol, probDic):
    sum = 0
    probSymbol = 0

    for s in probDic:
        if(s == symbol):
            probSymbol = probDic[s]
            break
        else:
            sum += probDic[s]

    sumWithSymbol = sum + probSymbol

    return [sum, sumWithSymbol]


# a dictionary with the cumulative probabilty of each symbol
cumulativeProbDic = {}

for symbol in probabilities:
    cumulativeProbDic[symbol] = getSumProb(symbol, probabilities)

# reshaping the image array where each block size is one element in the array
imgReshaped = np.reshape(img, (-1, blockSize))

def encode(sentence, prevLowerLimit, prevUpperLimit, probabilitiesDic, cumulativeProbDic, tag=''):
    
    if(sentence[0] == 0 and (not 0 in cumulativeProbDic)):
        return tag
    
    currentLowerLimit = prevLowerLimit + (prevUpperLimit - prevLowerLimit)*cumulativeProbDic[sentence[0]][0]
    currentUpperLimit = prevLowerLimit + (prevUpperLimit - prevLowerLimit)*cumulativeProbDic[sentence[0]][1]

    breakInfiniteLoop = 0
    while(currentUpperLimit <= 0.5 or currentLowerLimit >= 0.5):
        breakInfiniteLoop += 1
        if(breakInfiniteLoop == 50):
            break
        if(currentUpperLimit <= 0.5):
            tag += '0'
            currentUpperLimit = 2*currentUpperLimit
            currentLowerLimit = 2*currentLowerLimit
        else:
            tag += '1'
            currentUpperLimit = 2*(currentUpperLimit - 0.5)
            currentLowerLimit = 2*(currentLowerLimit - 0.5)

    if(len(sentence) == 1):
        tag += '1'
        return tag

    else:
        return encode(sentence[1:], currentLowerLimit, currentUpperLimit, probabilitiesDic, cumulativeProbDic, tag)


tags = []
print('⏳Encoding image, Please wait ...')
imgReshapedLength = len(imgReshaped)
for (index,block) in  enumerate( imgReshaped ):
    printProgressBar(index+1, imgReshapedLength, prefix = ' Progress:', suffix = 'Complete', length = 50)
    tags.append(encode(block, 0, 1, probabilities, cumulativeProbDic))

print('⏳Generating files to be used by decoder ...')
# Generating probabilities dictionary file
file = open("probabilities.pkl","wb")
pickle.dump(probabilities,file)
file.close()


# output the image info file
print(tags[0])
imgInfo = np.copy( [imgOriginal.shape] + [blockSize] + [zerosAtEnd] + tags)
np.save( 'imageInfo.npy',  imgInfo)
