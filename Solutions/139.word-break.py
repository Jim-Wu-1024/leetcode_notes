#
# @lc app=leetcode id=139 lang=python3
#
# [139] Word Break
#

# @lc code=start
from typing import List

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # Initialize the DP array where dp[j] is True if s[:j] can be segmented using words in wordDict
        dp = [False] * (len(s) + 1)
        dp[0] = True  # Base case: empty string can be segmented
        
        # Loop through each position in the string
        for j in range(1, len(s) + 1):
            # Check each word in wordDict to see if it can end at position j
            for word in wordDict:
                # Ensure the word length is not greater than j
                if j >= len(word):
                    # If dp[j - len(word)] is True and the substring matches the word, set dp[j] to True
                    if dp[j - len(word)] and s[j - len(word):j] == word:
                        dp[j] = True
                        break  # No need to check further if dp[j] is True
        
        # Return whether the entire string can be segmented
        return dp[-1]
        
# @lc code=end

