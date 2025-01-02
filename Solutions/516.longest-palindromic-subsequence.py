#
# @lc app=leetcode id=516 lang=python3
#
# [516] Longest Palindromic Subsequence
#

# @lc code=start
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        # Edge case: If the string length is 1, the longest palindromic subsequence is the string itself
        if len(s) == 1:
            return 1
        
        # Initialize a DP table with dimensions (len(s) x len(s))
        # dp[i][j] represents the length of the longest palindromic subsequence in s[i:j+1]
        dp = [[0] * len(s) for _ in range(len(s))]

        # Fill the DP table
        # Iterate from the end of the string to the beginning (bottom-up approach)
        for i in range(len(s) - 1, -1, -1):  # Corrected range for the outer loop
            dp[i][i] = 1  # A single character is always a palindrome of length 1
            for j in range(i + 1, len(s)):  # j > i to avoid redundant computations
                if s[i] == s[j]:
                    # If the characters match, extend the palindromic subsequence
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    # If they don't match, take the max of excluding one character
                    dp[i][j] = max(dp[i+1][j], dp[i][j-1])

        # The result is stored in dp[0][-1], which considers the entire string
        return dp[0][-1]

# @lc code=end

