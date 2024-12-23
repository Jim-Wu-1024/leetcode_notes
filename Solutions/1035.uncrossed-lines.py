#
# @lc app=leetcode id=1035 lang=python3
#
# [1035] Uncrossed Lines
#

# @lc code=start
from typing import List

class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        # Initialize DP table with dimensions (len(nums1)+1) x (len(nums2)+1)
        # dp[i][j] represents the maximum number of uncrossed lines between nums1[:i] and nums2[:j]
        dp = [[0] * (len(nums2) + 1) for _ in range(len(nums1) + 1)]

        result = 0  # Variable to store the maximum number of uncrossed lines

        # Fill the DP table
        for i in range(1, len(nums1) + 1):
            for j in range(1, len(nums2) + 1):
                if nums1[i-1] == nums2[j-1]:  # If the elements match
                    dp[i][j] = dp[i-1][j-1] + 1  # Extend the matching pair
                    result = max(result, dp[i][j])  # Update the result
                else:
                    # Otherwise, carry forward the maximum value by excluding one of the elements
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        # Return the maximum number of uncrossed lines
        return result
    
# @lc code=end

