from collections import defaultdict
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
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
    a1=TreeNode([1,2,3])
    a2= TreeNode([1, 2, 3])
    print(Solution().isSameTree(a1,a2))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
