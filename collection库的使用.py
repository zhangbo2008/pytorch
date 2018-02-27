#python 内置库collection的使用
import collections
from collections import *
#1.使用的计数器Counter
s = 'abcbcaccbbad'  
l = ['a','b','c','c','a','b','b']  
d = {'2': 3, '3': 2, '17': 2}  
# Counter 获取各元素的个数,返回字典  
print(Counter(s))   # Counter({'c': 4, 'b': 4, 'a': 3})  
print(Counter(l))   # Counter({'b': 3, 'a': 2, 'c': 2})  
print(Counter(d))   # Counter({3: 3, 2: 2, 17: 1})
m1 = Counter(s)  #按照元素出现的次数进行从高到低的排序,返回前int个元素的字典  
print(m1)                 # Counter({'c': 4, 'b': 4, 'a': 3, 'd': 1})  
print(m1.most_common(3))  # [('c', 4), ('b', 4), ('a', 3)]  
print (m1.elements()) #返回迭代器,表示m1里面的所有元素,
# update 和set集合的update一样,对集合进行并集更新  
u1 = Counter(s)  
u1.update('123a')  
print(u1)  # Counter({'a': 4, 'c': 4, 'b': 4, '1': 1, '3': 1, '2': 1})
# substract 和update类似，只是update是做加法，substract做减法,从另一个集合中减去本集合的元素，  
sub1 = 'which'  
sub2 = 'whatw'  
subset = Counter(sub1)  
print(subset)   # Counter({'h': 2, 'i': 1, 'c': 1, 'w': 1})  
subset.subtract(Counter(sub2))  
print(subset)   # Counter({'c': 1, 'i': 1, 'h': 1, 'a': -1, 't': -1, 'w': -1}) sub1中的h变为2，sub2中h为1,减完以后为1  
#2.双向队列deque
str1 = 'abc123cd'  
dq = deque(str1)  
print(dq)        # deque(['a', 'b', 'c', '1', '2', '3', 'c', 'd'])
dq = deque('abc123')  
dq.append('right')  
dq.appendleft('left')  
print(dq) # deque(['left', 'a', 'b', 'c', '1', '2', '3', 'right'])
#3.默认字典
dic = collections.defaultdict(dict)  
dic['k1'].update({'k2':'aaa'})  #比字典的update功能强
print(dic) 
#4.有序字典
# 定义有序字典  
import collections
print 
d1={}
d1['999a'] = 'A'
d1['cdsaf'] = 'B'
d1['1b23'] = 'C'
d1['134'] = '1'
d1['222'] = '2'
d1['9342399a'] = 'A'
d1['cd22222saf'] = 'B'
d1['1b3423'] = 'C'
d1['13232434'] = '1'
d1['22342'] = '2'
print (d1)
for k,v in d1.items():
    print (k,v)


d1 = collections.OrderedDict()
d1['999a'] = 'A'
d1['cdsaf'] = 'B'
d1['1b23'] = 'C'
d1['134'] = '1'
d1['222'] = '2'
for k,v in d1.items():
    print (k,v)








