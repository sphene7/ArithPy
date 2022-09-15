
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


if __name__ == '__main__':
    a1=TreeNode([1,2,3])
    a2= TreeNode([1, 2, 3])
    print(Solution().isSameTree(a1,a2))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
