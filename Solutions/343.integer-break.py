#
# @lc app=leetcode id=343 lang=python3
#
# [343] Integer Break
#

# @lc code=start
class Solution:
    def integerBreak(self, n: int) -> int:
        # Initialize dp array to store the maximum product for each integer up to n
        dp = [0] * (n + 1)
        
        # Base case: breaking 2 yields the maximum product of 1 (1 + 1)
        dp[2] = 1

        # Start filling dp array from 3 to n
        for i in range(3, n + 1):
            # Try to split the number i into two parts j and (i - j)
            for j in range(1, i // 2 + 1):
                # Calculate the maximum product by either:
                # 1. Not breaking (i - j) further: j * (i - j)
                # 2. Breaking (i - j) further using dp[i - j]
                dp[i] = max(j * (i - j), j * dp[i - j], dp[i])

        # The last element in dp array, dp[n], is the answer
        return dp[n]
    
# @lc code=end

