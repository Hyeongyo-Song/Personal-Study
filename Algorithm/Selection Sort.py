# Selection Sort를 구현하세요.

def Selection_Sort(arr):
    for i in range(len(arr)):
        mn = 999999999
        for j in range(i,len(arr)):
            if (arr[j] < mn): #i가 아닌 j
                mn = arr[j]
                idx = j
        arr[i],arr[idx] = arr[idx],arr[i] #들여쓰기 한칸 빼야 함.
    return arr

n = int(input("배열의 크기 지정 : "))
List = [x for x in range(n)]

for i in range(len(List)):
    print(i+1,"번째 수 :",end='')
    List[i] = int(input())

Sorted_List = Quick_Sort(List)

k = int(input("몇 번째로 작은 수를 찾으시겠습니까 ?"))
print(k,"번째로 작은 수는 ",Sorted_List[k-1], "입니다.")
