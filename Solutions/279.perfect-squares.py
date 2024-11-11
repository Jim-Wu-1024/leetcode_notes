#
# @lc app=leetcode id=279 lang=python3
#
# [279] Perfect Squares
#

# @lc code=start
class Solution:
    def numSquares(self, n: int) -> int:
        # Initialize the DP array with infinity, representing an initially unreachable sum
        dp = [float('inf')] * (n + 1)
        
        # Base case: 0 perfect squares are needed to achieve sum 0
        dp[0] = 0
        
        # Iterate over each integer from 1 up to n
        for i in range(1, n + 1):
            # Calculate the square of i and update the dp array for amounts from i**2 up to n
            square = i ** 2
            for j in range(square, n + 1):
                # Update dp[j] to be the minimum of its current value or dp[j - square] + 1
                # dp[j - square] + 1 represents adding one perfect square (i^2) to the solution for (j - square)
                dp[j] = min(dp[j], dp[j - square] + 1)
        
        # The final answer is the minimum number of perfect squares needed for sum n
        return dp[n]
    
# @lc code=end

