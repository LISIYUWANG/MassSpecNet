import itertools
import random

# 3个运算符号的排列
operation = list(itertools.product(['+', '-', '*', '/'], repeat=3))

# 随机抽4张
zh = [random.randint(1, 13) for i in range(4)]
print("抽到的牌：")
print(zh)
# 所有4个数无重复的排列
array = []
for nl in list(itertools.permutations(zh)):
    if nl not in array:
        array.append(nl)

# 组合
zh = [[str(nl[0]) + '*1.0'] + [ol[0]] + [str(nl[1]) + '*1.0'] + [ol[1]] + [str(nl[2]) + '*1.0'] + [ol[2]] + [
    str(nl[3]) + '*1.0'] for nl in array for ol in operation]

# 加括号
last = []
for come in zh:
    last.append(''.join(come[:2] + ['('] + come[2:5] + [')'] + come[5:]))
    last.append(''.join(come[:2] + ['('] + come[2:7] + [')']))
    last.append(''.join(['('] + come[:3] + [')'] + come[3:]))
    last.append(''.join(['('] + come[:5] + [')'] + come[5:]))
    last.append(''.join(['('] + come[:3] + [')'] + come[3:4] + ['('] + come[4:7] + [')']))
    last.append(''.join(come[:4] + ['('] + come[4:7] + [')']))

print("结果：")
times= 0
for may in last:
    #print("may1 ", may)
    try:
        if eval(may) == 24:  # 判断结果
            print(may.replace('*1.0', '') + '=24')
        else:
            times += 1
    except ZeroDivisionError as e:  #可能会有除0的情况
        times += 1
        continue
if times == len(last):
    print('无结果！')