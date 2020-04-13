import numpy as np
import cv2

img = cv2.imread('me.jpg',0)
print(img.shape)
img = img.flatten();
blockSize = 4

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


testDic = {
    'a': 0.7,
    'b': 0.1,
    'c': 0.2    
}

print("img before",img.shape)
imgReshaped = np.reshape(img, (-1,blockSize))
print("imgReshaped before",imgReshaped.shape)

def arithmeticCode(sentence, prevLowerLimit, prevUpperLimit, probabilitiesDic, cumulativeProbDic):    
    currentLowerLimit = prevLowerLimit + ( prevUpperLimit - prevLowerLimit )*cumulativeProbDic[ sentence[0] ][0]
    currentUpperLimit = prevLowerLimit + ( prevUpperLimit - prevLowerLimit )*cumulativeProbDic[ sentence[0] ][1]
    
    if( len(sentence) == 1 ):
        return ( currentLowerLimit + currentUpperLimit ) / 2 
    else:
        return arithmeticCode(sentence[1:], currentLowerLimit, currentUpperLimit, probabilitiesDic, cumulativeProbDic)
    
tags=[]
for x in imgReshaped:
    tags.append( arithmeticCode(x, 0, 1, probabilities, cumulativeProbDic) )
tag = arithmeticCode(imgReshaped[0], 0, 1, probabilities, cumulativeProbDic)

# print( arithmeticCode("abcb", 0, 1, testDic, {"a": [0,0.7], "b":[0.7,0.8], "c":[0.8,1] })) 
# print(len(tags))



# decoding function that takes the tag, probability dictionary and cumulative Probability Dic and return the decoded
def decode(tag, prevLowerLimit, prevUpperLimit, probabilitiesDic, cumulativeProbDic, terminator,decoded=[], counter=0): 
    counter+=1 # iterator to keep track of the number of function calls
    global decodedImg
    # terminate on this condition 
    if(counter == terminator):
        counter = 0
        return decodedImg
    
    # loop over all the symbols we have and get the range in which this tag exist
    for symbol in probabilitiesDic:
        currentUpperLimit = prevLowerLimit + ( prevUpperLimit - prevLowerLimit )*cumulativeProbDic[ symbol ][1]
        # if the tag exists in this range then this is the required symbol and we move on to get the next symbol
        if(tag <= currentUpperLimit ):
            decodedImg.append(symbol)
            currentLowerLimit = prevLowerLimit + ( prevUpperLimit - prevLowerLimit )*cumulativeProbDic[ symbol ][0]
            return decode(tag, currentLowerLimit, currentUpperLimit, probabilitiesDic, cumulativeProbDic,terminator, decoded, counter)            


# print ( decode(0.5565, 0, 1, testDic, {"a": [0,0.7], "b":[0.7,0.8], "c":[0.8,1] }, 5) )
# for tag in tags:
#     decode(0.5565, 0, 1, testDic, {"a": [0,0.7], "b":[0.7,0.8], "c":[0.8,1] }, 5)

# for tag in tags:
#     decodedImg.append( decode(tag, 0, 1, probabilities, cumulativeProbDic, blockSize+1, []) )
decodedImg = []
for tag in tags:
    decode(tag, 0, 1, probabilities, cumulativeProbDic, blockSize+1, [])

decodedImg=np.array(decodedImg[:len(decodedImg)-zerosAtEnd])

# decodedImg = np.reshape(decodedImg, (960,640)) #shoma.jpg
# decodedImg = np.reshape(decodedImg, (150,150)) #shoma640.jpg
# decodedImg = np.reshape(decodedImg, (4912,7360)) #large.jpg
# decodedImg = np.reshape(decodedImg, (3508,2480)) #test.jpg
decodedImg = np.reshape(decodedImg, (1589,1356)) #me.jog

cv2.imwrite('output.png',decodedImg)







# print(decodedImg[0])
# print(decode(tags[1], 0, 1, probabilities, cumulativeProbDic, blockSize+1) )

# print(imgReshaped[2])
# print(decode(tags[2], 0, 1, probabilities, cumulativeProbDic, blockSize+1))