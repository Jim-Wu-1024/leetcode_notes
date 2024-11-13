#
# @lc app=leetcode id=337 lang=python3
#
# [337] House Robber III
#

# @lc code=start
from typing import Optional, List

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        # Define a helper function for traversal that returns an array with two values:
        # - dp[0]: Maximum money if the current node is NOT robbed
        # - dp[1]: Maximum money if the current node IS robbed
        def traversal(cur: Optional[TreeNode]) -> List[int]:
            if not cur:
                # Base case: if the node is None, return [0, 0]
                return [0, 0]
            
            # Recursively solve for the left and right subtrees
            leftDp = traversal(cur.left)
            rightDp = traversal(cur.right)

            # dp[0]: If we do not rob this node, take the max of robbing or not robbing the children
            # dp[1]: If we rob this node, we cannot rob its children, so add its value to leftDp[0] and rightDp[0]
            return [
                max(leftDp[0], leftDp[1]) + max(rightDp[0], rightDp[1]),  # Max money without robbing this node
                cur.val + leftDp[0] + rightDp[0]  # Max money with robbing this node
            ]
        
        # Calculate the results for the root node
        dp = traversal(root)
        # Return the maximum money by choosing either to rob or not to rob the root node
        return max(dp[0], dp[1])
    
# @lc code=end

