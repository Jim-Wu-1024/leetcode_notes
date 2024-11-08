#
# @lc app=leetcode id=70 lang=python3
#
# [70] Climbing Stairs
#

# @lc code=start
class Solution:
    def climbStairs(self, n: int) -> int:
        # Base case: if there's only one step, there's only one way to climb it
        if n == 1:
            return 1
        
        # Initialize a DP array with two elements to store ways for the last two steps
        # dp[0] represents the number of ways to reach two steps before
        # dp[1] represents the number of ways to reach the last step
        dp = [1, 1]
        
        # Loop from step 2 up to n, calculating the number of ways to reach each step
        for _ in range(2, n + 1):
            # Calculate the number of ways to reach the current step
            # It's the sum of the ways to reach the previous step (dp[1])
            # and the step before that (dp[0])
            cur = dp[0] + dp[1]
            
            # Update the DP array for the next iteration
            # Shift the values forward: dp[1] becomes the new dp[0], and cur becomes the new dp[1]
            dp[0] = dp[1]
            dp[1] = cur
        
        # dp[1] now holds the number of ways to reach the nth step
        return dp[1]
# @lc code=end

