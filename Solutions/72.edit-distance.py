#
# @lc app=leetcode id=72 lang=python3
#
# [72] Edit Distance
#

# @lc code=start
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # Handle edge cases where one or both strings are empty
        if len(word1) == 0 or len(word2) == 0:
            # If either string is empty, the answer is the length of the other string
            return len(word1) if len(word2) == 0 else len(word2)
        
        # Initialize a DP table with dimensions (len(word1)+1) x (len(word2)+1)
        # dp[i][j] will represent the minimum number of operations required
        # to convert word1[:i] to word2[:j]
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        
        # Base case 1: If word2 is empty, delete all characters from word1
        for i in range(len(word1) + 1):
            dp[i][0] = i
        
        # Base case 2: If word1 is empty, insert all characters from word2
        for j in range(len(word2) + 1):
            dp[0][j] = j

        # Fill the DP table
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                # If the current characters match, no operation is needed for these characters
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # If characters do not match, consider three possible operations:
                    # 1. Delete from word1: dp[i-1][j] + 1
                    # 2. Insert into word1 (Delete from word2): dp[i][j-1] + 1
                    # 3. Replace the character: dp[i-1][j-1] + 1
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)

        # The result is stored in dp[len(word1)][len(word2)]
        return dp[-1][-1]

# @lc code=end

