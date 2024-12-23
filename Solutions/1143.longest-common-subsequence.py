#
# @lc app=leetcode id=1143 lang=python3
#
# [1143] Longest Common Subsequence
#

# @lc code=start
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # Initialize DP table with an extra row and column for the base case (0-indexed)
        # dp[i][j] represents the length of the LCS of text1[:i] and text2[:j]
        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]

        # Variable to track the result (maximum length of the LCS)
        result = 0

        # Populate the DP table
        for i in range(1, len(text1) + 1):
            for j in range(1, len(text2) + 1):
                if text1[i-1] == text2[j-1]:  # If the characters match
                    dp[i][j] = dp[i-1][j-1] + 1  # Extend the LCS
                    result = max(result, dp[i][j])  # Update the result if needed
                else:
                    # Otherwise, take the maximum LCS without one of the characters
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        # Return the final LCS length
        return result
        
# @lc code=end

