# 파이썬으로 합병 정렬을 구현하세요.

a = int(input("a>>>"))
b = int(input("b>>>"))
c = int(input("c>>>"))

def Merge_Sort(arr):
    if len(arr) < 2:
        return arr
    mid = len(arr) // 2
    left = Merge_Sort(arr[:mid])
    right = Merge_Sort(arr[mid:])
    L,R = 0,0
    merge = []
    while L<len(left) and R<len(right):
        if left[L] < right[R]:
            merge.append(left[L])
            L+=1
        else:
            merge.append(right[R])
            R+=1
        merge += left[L:]
        merge += right[R:]
        return merge

Sorted_List = Merge_Sort([a,b,c])
a = Sorted_List[2]
b = Sorted_List[1]
c = Sorted_List[0]
print("a :",a)
print("b :",b)
print("c :",c)
