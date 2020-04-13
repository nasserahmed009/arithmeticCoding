import numpy as np
import cv2

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

img = cv2.imread('screenshot.jpeg', 0)
imgOriginal = img
img = img.flatten()
blockSize = 16

probabilities = {}

totalPixels = len(img)
for colorDegree in img:
    if colorDegree in probabilities:
        probabilities[colorDegree] += 1
    else:
        probabilities[colorDegree] = 1



for colorDegree in probabilities:
    probabilities[colorDegree] = probabilities[colorDegree] / totalPixels
    
zerosAtEnd = blockSize - (blockSize if totalPixels %
                          blockSize == 0 else totalPixels % blockSize)
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

imgReshaped = np.reshape(img, (-1, blockSize))


def encode(sentence, prevLowerLimit, prevUpperLimit, probabilitiesDic, cumulativeProbDic, tag=''):

    if(sentence[0] == 0 and (not 0 in cumulativeProbDic)):
        return tag
    currentLowerLimit = prevLowerLimit + \
        (prevUpperLimit - prevLowerLimit)*cumulativeProbDic[sentence[0]][0]
    currentUpperLimit = prevLowerLimit + \
        (prevUpperLimit - prevLowerLimit)*cumulativeProbDic[sentence[0]][1]

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

print('✅ Encoding done successfully')


def binToFloat(binary):
    if(len(binary) == 0):
        return 0

    number = 0

    for index in range(0, len(binary)):
        number += int(binary[index])*(2**(-(index+1)))
    return number



# # decoding function that takes the tag, probability dictionary and cumulative Probability Dic and return the decoded
def decode(tag, prevLowerLimit, prevUpperLimit, probabilitiesDic, cumulativeProbDic, terminator):
    # iterator to keep track of the number of function calls
    global decodedImg
    global counter

    # terminate on this condition
    if(counter == terminator):
        return

    currentLowerLimit = prevLowerLimit
    currentUpperLimit = prevUpperLimit

    while(currentUpperLimit <= 0.5 or currentLowerLimit >= 0.5):
        if(currentUpperLimit <= 0.5):
            tag = tag[1:]
            currentUpperLimit = 2*currentUpperLimit
            currentLowerLimit = 2*currentLowerLimit
        else:
            tag = tag[1:]
            currentUpperLimit = 2*(currentUpperLimit - 0.5)
            currentLowerLimit = 2*(currentLowerLimit - 0.5)

    prevLowerLimit = currentLowerLimit
    prevUpperLimit = currentUpperLimit
    currentRange = prevUpperLimit - prevLowerLimit

    tagDecimal = binToFloat(tag)
    # loop over all the symbols we have and get the range in which this tag exist
    for symbol in probabilitiesDic:
        currentUpperLimit = prevLowerLimit + (currentRange)*cumulativeProbDic[symbol][1]

        # if the tag exists in this range then this is the required symbol and we move on to get the next symbol
        if(tagDecimal<=currentUpperLimit):
            currentLowerLimit = prevLowerLimit + (currentRange)*cumulativeProbDic[symbol][0]
            counter += 1
            decodedImg.append(symbol)
            return decode(tag, currentLowerLimit, currentUpperLimit, probabilitiesDic, cumulativeProbDic, terminator)

decodedImg = []

print('⏳Decoding image, Please wait ...')
pixelsErrors = 0
counter = 0
tagsLength = len(tags)
for (index,tag) in enumerate(tags):
    printProgressBar(index+1, tagsLength, prefix = ' Progress:', suffix = 'Complete', length = 50)
    decode(tag, 0, 1, probabilities, cumulativeProbDic, blockSize)
    
    if(counter != blockSize):
        color = 0 
        decodedImgLength = len(decodedImg)

        for i in range (len(decodedImg)-16, len(decodedImg)):
            color += decodedImg[i]
        
        color = int(color/15)
        pixelsErrors += (blockSize-counter)
        decodedImg = decodedImg[:decodedImgLength-4]
        for index in range(0, blockSize-counter+4):
            decodedImg.append(color)
    counter = 0

print('✅ Decoding done successfully')
print('⚠️ Error percentage', round( (pixelsErrors/totalPixels)*100, 2), '%')

decodedImg = np.array(decodedImg[:len(decodedImg)-zerosAtEnd])
decodedImg = np.reshape(decodedImg, imgOriginal.shape)

cv2.imwrite('output.jpg', decodedImg)
print('✅ Image saved as "output.jpg"')

