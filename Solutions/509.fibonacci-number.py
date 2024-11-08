#
# @lc app=leetcode id=509 lang=python3
#
# [509] Fibonacci Number
#

# @lc code=start
class Solution:
    def fib(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        
        # Initialize a DP array with size 2 to store only the last two values
        dp = [0, 1]
        
        # Iteratively fill up DP values, reusing the 2-element array
        for i in range(2, n + 1):
            # Compute the next Fibonacci number by summing the two last stored values
            current = dp[0] + dp[1]
            
            # Update the array: shift the values to the left
            dp[0] = dp[1]
            dp[1] = current
        
        # Return the last computed Fibonacci number, which is in dp[1]
        return dp[1]
    
# @lc code=end

