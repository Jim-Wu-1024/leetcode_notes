#
# @lc app=leetcode id=455 lang=python3
#
# [455] Assign Cookies
#

# @lc code=start
from typing import List

class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        # Sort the greed factors of children in ascending order
        g.sort()  
        # Sort the sizes of cookies in ascending order
        s.sort()  
        
        # Start with the largest cookie index
        index = len(s) - 1  
        result = 0  # Initialize the result to count content children
        
        # Iterate over the children starting from the most greedy to the least
        for i in range(len(g) - 1, -1, -1):  
            # Check if there is a cookie available and if it can satisfy the current child's greed
            if index >= 0 and s[index] >= g[i]:  
                result += 1  # Increment the result as the child is content
                index -= 1  # Move to the next largest available cookie
        
        return result  # Return the total number of content children
        
# @lc code=end

