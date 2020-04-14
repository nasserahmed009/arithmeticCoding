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
    
imgOriginal = img
img = img.flatten();

validBlockSize = False
while not validBlockSize:
    try:
        blockSize = input("⚙️ Enter block size : ")
        blockSize = int(blockSize)
        validBlockSize = True 
    except:
        print("❌ This blocksize isn't valid, Please enter an intger")


validDataType = False
while not validDataType:
    try:
        print("⚙️ Choose the preferred datatype [1,2 or 3] : ")
        print("1. float16")
        print("2. float32")
        print("3. float64")

        dataType = input("Datatype : ")
        dataType = int(dataType)
        if( dataType > 3 or dataType < 1 ):
            raise Exception("Can't find this dataType")
        validDataType = True 
    except:
        print("❌ This datatype is not valid please enter 1, 2 or 3")

    
if(dataType == 1):
    dataType = 'float16'
elif(dataType == 2):
    dataType = 'float32'
else:
    dataType = 'float64'


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
  probabilities[colorDegree] = probabilities[colorDegree] / totalPixels 


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
        return ( currentLowerLimit + currentUpperLimit ) / 2
    else:
        return arithmeticCode(sentence[1:], currentLowerLimit, currentUpperLimit, probabilitiesDic, cumulativeProbDic)


tags=np.zeros(len(imgReshaped), dtype=dataType)
imgReshapedLength = len(imgReshaped)
for index, block in enumerate(imgReshaped):
    printProgressBar(index+1, imgReshapedLength, prefix = ' Progress:', suffix = 'Complete', length = 50)
    tags[index] = arithmeticCode(block, 0, 1, probabilities, cumulativeProbDic) 


print('⏳Generating files to be used by decoder ...')
# Generating probabilities dictionary file
file = open("probabilities.pkl","wb")
pickle.dump(probabilities,file)
file.close()


# output the image info file
# imgInfo = np.array( [imgOriginal.shape] + [blockSize] + [zerosAtEnd] + tags  )
np.save( 'imageInfo.npy', np.array([imgOriginal.shape] + [blockSize] + [zerosAtEnd]) )
np.save( 'tags.npy',  tags)

