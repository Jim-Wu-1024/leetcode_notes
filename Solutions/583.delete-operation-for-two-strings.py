#
# @lc app=leetcode id=583 lang=python3
#
# [583] Delete Operation for Two Strings
#

# @lc code=start
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # Initialize a DP table with dimensions (len(word1)+1) x (len(word2)+1)
        # dp[i][j] will store the minimum number of deletions required
        # to make word1[:i] and word2[:j] identical
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]

        # Base case 1: If one string is empty, the only option is to delete all characters
        # from the other string
        for i in range(len(word1) + 1):
            dp[i][0] = i  # Deleting all characters from word1[:i]
        for j in range(len(word2) + 1):
            dp[0][j] = j  # Deleting all characters from word2[:j]

        # Fill the DP table
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                # If the characters match, no additional deletion is required for these characters
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # Take the result from the previous diagonal cell
                else:
                    # If the characters do not match, consider three possible options:
                    # 1. Delete the character from word1: dp[i-1][j] + 1
                    # 2. Delete the character from word2: dp[i][j-1] + 1
                    # 3. Delete the characters from both word1 and word2: dp[i-1][j-1] + 2
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 2)

        # The result is stored in the bottom-right corner of the DP table
        return dp[-1][-1]
    
# @lc code=end

