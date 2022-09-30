from collections import defaultdict
from typing import List, Optional

INT_MAX=2**32-1
INT_MIN=-2**32

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# class Automaton:
#     def isMatch(self, s: str, p: str) -> bool:
#
#     def __init__(self):
#         self.state = 'start'
#         self.sign = 1
#         self.ans = 0
#         self.table = {
#             'start': ['start', 'signed', 'in_number', 'end'],
#             'signed': ['end', 'end', 'in_number', 'end'],
#             'in_number': ['end', 'end', 'in_number', 'end'],
#             'end': ['end', 'end', 'end', 'end'],
#         }
#
#     def get_col(self, c):
#         if c.isspace():
#             return 0
#         if c == '+' or c == '-':
#             return 1
#         if c.isdigit():
#             return 2
#         return 3
#
#     def get(self, c):
#         self.state = self.table[self.state][self.get_col(c)]
#         if self.state == 'in_number':
#             self.ans = self.ans * 10 + int(c)
#             self.ans = min(self.ans, INT_MAX) if self.sign == 1 else min(self.ans, -INT_MIN)
#         elif self.state == 'signed':
#             self.sign = 1 if c == '+' else -1

class Solution:
    # def strStr(self, haystack: str, needle: str) -> int:
    #     def calShiftMat(st):
    #         dic={}
    #         for i in range(len(st)-1,-1,-1):
    #             if not dic.get(st[i]):
    #                 dic[st[i]]=len(st)-i
    #         dic["ot"]=len(st)+1
    #         return dic
    #     if len(needle)>len(haystack):return -1
    #     if needle==" ":return 0
    #     dic=calShiftMat(needle)
    #     idx=0
    #     while idx+len(needle)<=len(haystack):
    #         str_cut=haystack[idx:idx+len(needle)]
    #         if str_cut==needle:
    #             return idx
    #         else:
    #             if idx+len(needle)>=len(haystack):
    #                 return -1
    #             cur_c=haystack[idx+len(needle)]
    #             if dic.get(cur_c):
    #                 idx+=dic[cur_c]
    #             else:
    #                 idx+=dic["ot"]
    #         return -1 if idx+len(needle)>=len(haystack) else idx


    # def removeElement(self, nums: List[int], val: int) -> int:  #####27
    #     a,b=0,0
    #     while a<len(nums):
    #         if nums[a]!=val:
    #             nums[b]=nums[a]
    #             b+=1
    #         a+=1
    #     return b


    # def removeDuplicates(self, nums: List[int]) -> int:   #####26
    #     n=len(nums);j=0
    #     for i in range(n):
    #         if nums[i]!=nums[j]:
    #             j+=1
    #             nums[j]=nums[i]
    #     return j+1

    # def swapPairs(self, head):
    #     if not (head ):
    #         return head
    #     p=ListNode(-1)
    #     cur,head,stack=head,p,[]
    #     while cur and cur.next:
    #         _,_=stack.append(cur),stack.append(cur.next)
    #         cur=cur.next.next
    #         p.next=stack.pop()
    #         p.next.next=stack.pop()
    #         p=p.next.next
    #     if cur:  #前面的while只能在偶数里走，这个是判断他是不是奇数哒
    #         p.next=cur
    #     else:
    #         p.next=None
    #     return head.next


    # def mergeKLists(self, lists: List[ListNode]) -> ListNode:    #####23
    #     if not lists: return
    #     n=len(lists)
    #     return self.merge(lists,0,n-1)
    # def merge(self,lists,left,right):
    #     if left==right:
    #         return lists[left]
    #     mid=left+(right-left)//2
    #     l1=self.merge(lists,left,mid)
    #     l2=self.merge(lists,mid+1,right)
    #     return self.mergeTlists(l1,l2)
    # def mergeTlists(self,l1,l2):
    #     if not l1:return l2
    #     if not l2: return l1
    #     if l1.val<=l2.val:
    #         l1.next=self.mergeTlists(l1.next,l2)
    #         return l1
    #     else:
    #         l2.next=self.mergeTlists(l2.next,l1)
    #         return l2

    # def generateParenthesis(self, n: int) -> List[str]:
    #     res,cur_str=[],''
    #     def dfs(cur_str,left,right,n):
    #         if left ==n and right==n:
    #             res.append(cur_str)
    #         if left<right:    #右括号用太多啦
    #             return
    #         if left<n:
    #             dfs(cur_str+'(',left+1,right,n)
    #         if right<n:
    #             dfs(cur_str+')',left,right+1,n)
    #     dfs(cur_str,0,0,n)
    #     return res


    # def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    #     if not l1: return l2  # 终止条件，直到两个链表都空
    #     if not l2: return l1
    #     if l1.val <= l2.val:  # 递归调用
    #         l1.next = self.mergeTwoLists(l1.next, l2)
    #         return l1
    #     else:
    #         l2.next = self.mergeTwoLists(l1, l2.next)
    #         return l2

    # def isValid(self, s: str) -> bool:     ######20
    #     dic={'[':']','(':')','{':'}','?':'?'}
    #     stack=['?']
    #     for c in s:
    #         if c in dic:
    #             stack.append(c)
    #         elif dic[stack.pop()]!=c: return False
    #     return len(stack)==1

    # def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    # def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:   #####19
    #     # hT=ListNode(None)
    #     # hT.next=head
    #     hT = ListNode(0, head)
    #     a_ptr, b_ptr = hT, hT
    #     # for i in range(n):
    #     for i in range(n + 1):
    #         b_ptr = b_ptr.next
    #     while b_ptr != None:
    #         b_ptr = b_ptr.next
    #         a_ptr = a_ptr.next
    #     a_ptr.next = a_ptr.next.next
    #     return hT.next

    # def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
    #     res,n=[],len(nums)
    #     if not nums or n<4:
    #         return res
    #     nums.sort()
    #     # for a in range n-3:
    #     for a in range (n-3):
    #         # if nums[a-1]==nums[a]:
    #         if nums[a-1]==nums[a] and a>0:   #?????????这个a>0什么用啊
    #             continue
    #             ##缩进
    #         for b in range(a+1,n-2):
    #             if b>a+1 and nums[b-1]==nums[b]:
    #                 continue
    #             c=b+1;d=n-1
    #             while c<d:
    #                 sum=nums[a]+nums[b]+nums[c]+nums[d]
    #                 if sum==target:
    #                     # res.append(nums[a],nums[b],nums[c],nums[d])
    #                     res.append([nums[a],nums[b],nums[c],nums[d]])
    #                     #注意缩进
    #                     while c<d and nums[c]==nums[c+1]:
    #                         c+=1
    #                     while c<d and nums[d]==nums[d-1]:
    #                         d-=1
    #                     c+=1
    #                     d-=1
    #                 elif sum<target:
    #                     c+=1
    #                 else:
    #                     d-=1
    #     return res


    # def letterCombinations(self, digits: str) -> List[str]:
    #     res = []
    #     if not digits: return []
    #     phone = {'2':['a','b','c'],
    #              '3': ['d', 'e', 'f'],
    #              '4': ['g', 'h', 'i'],
    #              '5': ['j', 'k', 'l'],
    #              '6': ['m', 'n', 'o'],
    #              '7': ['p', 'q', 'r', 's'],
    #              '8': ['t', 'u', 'v'],
    #              '9': ['w', 'x', 'y', 'z']}
    #     def backtrack(resT,next):  #resT存字符，next存下一个数字
    #         if len(next)==0:
    #             res.append(resT)
    #         else:
    #             for i in phone[next[0]]:     #i是字母哦
    #                 backtrack(resT+i,next[1:])
    #     backtrack('',digits)
    #     return res

    # def threeSum(self, nums: List[int]) -> List[List[int]]:    #16
    #     n=len(nums)
    #     res=[]
    #     if (not nums or n<3):
    #         return []
    #     nums.sort()
    #     #for i in range nums:
    #     for i in range(n):
    #         if nums[i]>0:
    #             return res
    #         if(i>0 and nums[i]==nums[i-1]):
    #             continue
    #         L=i+1;R=n-1
    #         while L<R:
    #             if(nums[i]+nums[L]+nums[R]==0):
    #                 # res.append(nums[i],nums[L],nums[R])  是的，否则就是超出参数
    #                 res.append([nums[i],nums[L],nums[R]])
    #                 # while(nums[L]==nums[L+1])
    #                 while(L<R and nums[L]==nums[L+1]):  #防一下多个并排加错
    #                     L=L+1
    #                 while(L<R and nums[R]==nums[R-1]):
    #                     R=R-1
    #                 L+=1;R-=1
    #             ###你这个笨蛋准备不写elif
    #             elif(nums[i]+nums[L]+nums[R]>0):
    #                 R=R-1
    #             else:
    #                 L+=1
    #
    #         # return res
    #     return res


    # def longestCommonPrefix(self, s: List[str]) -> str:     ##########14
    #     if not s:
    #         return ""
    #     res = s[0]     #第一个单词
    #     i = 1
    #     while i < len(s):   #总len(s)个单词
    #         while s[i].find(res) :
    #             res = res[0:len(res) - 1]
    #         i += 1
    #     return res


    # def romanToInt(self,s:str)->int:   #######13 Roman To index
    #     d = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, 'L': 50, 'XC': 90, 'C': 100, 'CD': 400, 'D': 500,
    #          'CM': 900, 'M': 1000}
    #     tempI=0;tempN=0
    #     while tempI<=len(s)-1:
    #         if s[tempI:tempI+2] in d:
    #             #tempN=d.values(s[tempI:tempI+2])
    #             #tempN=d.get(s[tempI:tempI+2])  #神TM等于....
    #             tempN+=d.get(s[tempI:tempI+2])
    #             tempI+=2
    #         else:
    #             tempN += d.get(s[tempI])
    #             tempI += 1
    #     return tempN

    # def intToRoman(self, num: int) -> str:    #####12 Roman
    #     nums = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    #     romans = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    #     index =0
    #     res=''
    #     while index<=12:
    #         while num>=nums[index]:
    #             #res.appends
    #             res+=romans[index]
    #             num-=nums[index]
    #         index+=1
    #     return res

    # def maxArea(self, height: List[int]) -> int:   #####11  Most Water
    #     i,j,res,=0,len(height)-1,0    #数组..
    #     while i<j:
    #         if height[i]<height[j]:
    #             # res=max(res,height[i+1]*(j-i))
    #             res=max(res,height[i]*(j-i))
    #             i+=1
    #         else:
    #             #res=max(res,height[j-1]*(j-i))
    #             res=max(res,height[j]*(j-i))
    #             j-=1
    #     return res

    # def isMatch(self, s: str, p: str) -> bool:    ######10
    #     len_s=len(s);len_p=len(p)
    #     #dp=[False*(len_p+1) for _ in range(len_s+1)]
    #     dp=[[False]*(len_p+1) for _ in range(len_s+1)]
    #     #为什么是+1呢？
    #     dp[0][0]=True
    #     for j in range(1,len_p+1):   #只是第一P的情况
    #         if p[j-1]=='*':
    #             dp[0][j]=dp[0][j-2]
    #     for i in range(1,len_s+1):
    #         for j in range (1,len_p+1):
    #             if p[j-1] in {s[i-1],'.'}:
    #                 #dp[i][j]=dp[i][j-2]
    #                 dp[i][j]=dp[i-1][j-1]
    #             elif p[j-1]=='*':       #这几行？？？？？
    #                 if p[j-2] in {s[i-1],'.'}:
    #                     dp[i][j]=dp[i][j-2]or dp[i-1][j]
    #                     #dp[i][j]=dp[i][j-2] or dp
    #                 else:
    #                     dp[i][j]=dp[i][j-2]
    #     return dp[len_s][len_p]

                    #if p[j-2]=='.'

    # def myAtoi(self, str: str) -> int:
    #     automaton = Automaton()
    #     for c in str:
    #         automaton.get(c)
    #     return automaton.sign * automaton.ans


    # def reverse(self, x: int) -> int:     ###### 7
    #     str_x=str(x)
    #     #if len(str_x)
    #     if str_x[0]!="-":
    #         str_x=str_x[::-1]
    #     else:
    #         #str_x[0]="-"    //不能这么赋值
    #         #str 也不能append
    #         a=str_x[:0:-1]      #这是消去第一位，惊喜吗^-^
    #         str_x="-"+a
    #     int_x=int(str_x)
    #     if -2147483648 < int_x < 2147483647:return int_x
    #     else: return 0

    # def convert(self, s: str, numRows: int) -> str:
    #     if numRows<2:return s
    #     #res=[""for _ in range(s)]
    #     res=[""for _ in range(numRows)]
    #     i,flag=0,-1
    #     for c in s:
    #         res[i]+=c
    #         if i==0 or i==numRows-1:
    #             flag=-flag
    #         i+=flag
    #     return "".join(res)

    # def longestPalindrome(self, s: str) -> str:  ###5
    #     n=len(s)
    #     if n<2:
    #         return s
    #     max_l=1;begin=0
    #     dp=[[False]*n for _ in range(n)]
    #     for i in range(n):
    #         dp[i][i] =True
    #     for L in range(2,n+1):   #下标
    #         for i in range(n):
    #             j=L+i-1
    #             if j>=n:
    #                 break
    #             if s[i]!=s[j]:
    #                 dp[i][j]=False
    #             else:
    #                 if j-i<3:
    #                     dp[i][j]=True
    #                 else:
    #                     dp[i][j]=dp[i+1][j-1]
    #             if dp[i][j] and j-i+1>max_l:   #因为第一个也算一个元素呀
    #                 max_l=j-i+1
    #                 begin=i
    #     return s[begin:begin+max_l]

    # def lengthOfLongestSubstring(self, s: str) -> int:   #####03
    #     if not s:
    #         return 0
    #     left=0
    #     lookup=set()
    #     n=len(s)
    #     max_len=0;cur_len=0
    #     for i in range( n):
    #         cur_len+=1
    #         while s[i] in lookup:
    #             lookup.remove(s[left])   #字符串呀
    #             left+=1
    #             cur_len-=1
    #         if cur_len>max_len:
    #             max_len=cur_len
    #         lookup.add(s[i])
    #     return max_len

    #     def addTwoNumbers1(self, l1, l2):
    #         re = ListNode(0)
    #         r = re
    #         carry = 0
    #         while (l1 or l2):
    #             x = l1.val if l1 else 0
    #             y = l2.val if l2 else 0
    #             s = carry + x + y
    #             carry = s // 10
    #             r.next = ListNode(s % 10)
    #             r = r.next
    #             if (l1 != None): l1 = l1.next
    #             if (l2 != None): l2 = l2.next
    #         if (carry > 0):
    #             r.next = ListNode(1)
    #         return re.next
    #     def addTwoNumbers(self, l1, l2):  #####2 ?
    #         re = ListNode(0)
    #         r = re
    #         TmpD = 0
    #         while (l1 or l2):
    #             a = l1.val if l1 else 0
    #             b = l2.val if l2 else 0
    #             Tmp = TmpD + a + b
    #             TmpD = Tmp // 10
    #             r.next = ListNode(Tmp % 10)
    #             r = r.next
    #         if (TmpD > 0):
    #             r.next = ListNode(1)
    #         return re.next

    # def twoSum(self, nums: List[int], target: int) -> List[int]:   ######1
    #     n=len(nums)
    #     #for i in range[n]:
    #     for i in range(n):
    #         #for j in range[n-i]
    #         for j in range(i+1,n):
    #             if nums[i]+nums[j]==target:
    #                 #-> List[int]:
    #                 return [i,j]
    #     return []


    def twoSum(self, nums: List[int], target: int) -> List[int]:  #####1 ->Hash
         hashtable=dict()
         for i,num in enumerate(nums):
             if target-num in hashtable:
                 return[hashtable[target-num],i]
             #hashtable[num]=i
             #hashtable[num[i]]=i
             hashtable[nums[i]]=i
         return []

    #     hashtable=dict()
    #     for i,num in enumerate(nums):  #枚举
    #         if target-num in hashtable:
    #             return [hashtable[target-num],i]  #返回下标呀
    #         hashtable[nums[i]]=i   #返回下标
    #     return []


    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        elif not p or not q:
            return False
        elif p.val != q.val:
            return False
        else:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

        def addBinary(self, a: str, b: str) -> str:
            return '{0:b}'.format(int(a,2)+int(b,2))  #二进制a,b,相加

        def getPermutation(self, n:int, k:int)-> str:   #60
            #used=[False for _ in range[n+1]]
            def dfs(n,k,id,path):
                if id==n:
                    return
                cnt=f[n-1-id]
                for i in range(1,n+1):
                    if used[i]:
                        continue
                    if cnt<k:
                        k-=cnt
                        continue
                    path.append(i)
                    used[i]=True
                    dfs(n,k,id+1,path)
                    return

            if n==0:
                return ""
            used = [False for _ in range(n + 1)]
            path = []
            f = [1 for _ in range(n + 1)]

            for i in range(2, n + 1):
                f[i] = f[i - 1] * i

            dfs(n,k,0,path)
            return ''.join([str(num) for num in path])


        def minWindow(self, s: str, t: str) -> str:   #76 To change
            if len(s)<len(t):
                return ""
            hs,ht=defaultdict(int),defaultdict(int)
            for char in t:
                ht[char]+=1

            res = ""
            left,right=0,0

            cnt=0
            while right<len(s):
                hs[s[right]]+=1
                if hs[s[right]]<=hs[s[right]]:
                    cnt+=1
                while left<right and hs[s[left]]>ht[s[left]]:
                    hs[s[left]]-=1
                    left+=1
                if cnt==len(t):
                    if not res or right-left+1<len(res):
                        res=s[left:right+1]
                right+=1
            return res


    # def getPermutation(self, n: int, k: int) -> str:
    #     def dfs(n, k, index, path):
    #         if index == n:
    #             return
    #         cnt = factorial[n - 1 - index]
    #         for i in range(1, n + 1):
    #             if used[i]:
    #                 continue
    #             if cnt < k:
    #                 k -= cnt
    #                 continue
    #             path.append(i)
    #             used[i] = True
    #             dfs(n, k, index + 1, path)
    #             # 注意：这里要加 return，后面的数没有必要遍历去尝试了
    #             return
    #
    #     if n == 0:
    #         return ""
    #
    #     used = [False for _ in range(n + 1)]
    #     path = []
    #     factorial = [1 for _ in range(n + 1)]
    #     for i in range(2, n + 1):
    #         factorial[i] = factorial[i - 1] * i
    #
    #     dfs(n, k, 0, path)
    #     return ''.join([str(num) for num in path])





if __name__ == '__main__':
    # a1=TreeNode([1,2,3])
    # a2= TreeNode([1, 2, 3])
    # print(Solution().isSameTree(a1,a2))
    #print(Solution.convert("PAYPALISHIRING",3))
    #print(Solution().convert("PAYPALISHIRING",3))
    print(Solution().fourSum([2,2,2,2,2],8))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
