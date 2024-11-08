#
# @lc app=leetcode id=62 lang=python3
#
# [62] Unique Paths
#

# @lc code=start
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n  # Initialize a 1D dp array with 1s, representing the first row
        
        for _ in range(1, m):  # Start from the second row
            for col in range(1, n):  # Start from the second column
                dp[col] += dp[col - 1]  # Update the current cell with the sum of the cell to the left and itself

        return dp[-1]  # The last element contains the number of unique paths
    
# @lc code=end

