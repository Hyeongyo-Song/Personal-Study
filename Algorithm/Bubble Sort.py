def Bubble(numbers):
    for i in range(len(numbers),1,-1):
        print(numbers)
        for j in range(i-1):
            if numbers[j] > numbers[j+1]:
                numbers[j], numbers[j+1] = numbers[j+1], numbers[j]

import random

List = [0 for i in range(10)]

for i in range(0,10):
    List[i] = random.randint(1,100)

Bubble(List)
