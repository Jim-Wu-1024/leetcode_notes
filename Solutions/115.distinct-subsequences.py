#
# @lc app=leetcode id=115 lang=python3
#
# [115] Distinct Subsequences
#

# @lc code=start
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        # If s is shorter than t, it's impossible to form t
        if len(s) < len(t):
            return 0
        
        # Initialize DP table with dimensions (len(s)+1) x (len(t)+1)
        # dp[i][j] represents the number of ways to form t[:j] as a subsequence of s[:i]
        dp = [[0] * (len(t) + 1) for _ in range(len(s) + 1)]
        
        # Base case: An empty string t can be formed by deleting all characters of s
        for i in range(len(s) + 1):
            dp[i][0] = 1

        # Fill the DP table
        for i in range(1, len(s) + 1):
            for j in range(1, len(t) + 1):
                if s[i-1] == t[j-1]:  # Characters match
                    # Option 1: Use this match
                    # Option 2: Skip this character in s
                    dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
                else:  # Characters don't match
                    # Skip this character in s
                    dp[i][j] = dp[i-1][j]

        # The result is stored in dp[len(s)][len(t)]
        return dp[-1][-1]
    
# @lc code=end

