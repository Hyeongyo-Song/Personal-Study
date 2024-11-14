# Quick Sort를 구현하세요.

def Quick_Sort(arr):
    if len(arr) < 2:
        return arr
    pivot = arr[0]
    less = [x for x in arr[1:] if x<pivot]
    grate = [x for x in arr[1:] if x>pivot]
    less = Quick_Sort(less)
    grate = Quick_Sort(grate)
    return less + [pivot] + grate

n = int(input("배열의 크기 지정 : "))
List = [x for x in range(n)]

for i in range(len(List)):
    print(i+1,"번째 수 :",end='')
    List[i] = int(input())

Sorted_List = Quick_Sort(List)

k = int(input("몇 번째로 작은 수를 찾으시겠습니까 ?"))
print(k,"번째로 작은 수는 ",Sorted_List[k-1], "입니다.")
