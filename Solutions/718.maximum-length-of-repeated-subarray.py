#
# @lc app=leetcode id=718 lang=python3
#
# [718] Maximum Length of Repeated Subarray
#

# @lc code=start
from typing import List

class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        # Initialize the DP table
        # dp[i][j] will store the length of the longest common subarray ending at nums1[i] and nums2[j]
        dp = [[0] * len(nums2) for _ in range(len(nums1))]

        result = 0  # Variable to track the maximum length of repeated subarray
        
        # Fill the first column of the DP table
        for i in range(len(nums1)):
            dp[i][0] = 1 if nums1[i] == nums2[0] else 0
            result = max(result, dp[i][0])  # Update the maximum result
        
        # Fill the first row of the DP table
        for j in range(len(nums2)):
            dp[0][j] = 1 if nums2[j] == nums1[0] else 0
            result = max(result, dp[0][j])  # Update the maximum result

        # Populate the rest of the DP table
        for i in range(1, len(nums1)):
            for j in range(1, len(nums2)):
                if nums1[i] == nums2[j]:  # If characters match, extend the common subarray
                    dp[i][j] = dp[i-1][j-1] + 1
                    result = max(result, dp[i][j])  # Update the maximum result

        return result  # Return the maximum length of the repeated subarray
    
# @lc code=end

