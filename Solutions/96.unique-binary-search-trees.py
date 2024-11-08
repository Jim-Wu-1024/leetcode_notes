#
# @lc app=leetcode id=96 lang=python3
#
# [96] Unique Binary Search Trees
#

# @lc code=start
class Solution:
    def numTrees(self, n: int) -> int:
        # Handle the base case for n=1 directly
        if n == 1:
            return 1

        # Initialize dp array where dp[i] represents the number of unique BSTs with i nodes
        dp = [0] * (n + 1)
        
        # Base cases:
        dp[0] = 1  # Empty tree (0 nodes) has one unique structure
        dp[1] = 1  # One node has only one unique structure
        dp[2] = 2  # Two nodes can be arranged in two unique BST structures
        
        # Fill dp array for each number of nodes from 3 to n
        for i in range(3, n + 1):
            # Calculate dp[i] by summing the number of unique BSTs for each possible root position
            for j in range(1, i + 1):
                # dp[j-1] represents left subtree options, dp[i-j] represents right subtree options
                dp[i] += dp[j - 1] * dp[i - j]
        
        # The result is the number of unique BSTs that can be formed with n nodes
        return dp[n]
    
# @lc code=end

