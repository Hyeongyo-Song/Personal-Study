def Insert(numbers):
    for i in range(1, len(numbers)):
        t = numbers[i]
        print(numbers)
        for j in range(i,0,-1):
            if numbers[j-1] > t:
                numbers[j], numbers[j-1] = numbers[j-1], numbers[j]
            else:
                break

import random

List = [0 for i in range(10)]

for i in range(0,10):
    List[i] = random.randint(1,100)

Insert(List)
