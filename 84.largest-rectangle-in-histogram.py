#
# @lc app=leetcode id=84 lang=python3
#
# [84] Largest Rectangle in Histogram
#

# @lc code=start
from typing import List

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # Initialize a stack to keep track of indices of heights
        stack = []
        max_area = 0
        
        # Append a zero-height bar to ensure all elements in the stack get processed
        # [4, 6, 8]
        heights.append(0)
        
        for i, h in enumerate(heights):
            # Ensure the stack maintains a non-decreasing order of heights
            while stack and heights[stack[-1]] > h:
                # Pop the top element (height index)
                height = heights[stack.pop()]
                
                # Calculate the width
                # If the stack is empty, the width is the current index i
                # This happens when there are no smaller heights to the left,
                # meaning the rectangle extends from index 0 to index i.
                width = i if not stack else i - stack[-1] - 1
                
                # Update the maximum area
                max_area = max(max_area, height * width)
            
            # Push the current index onto the stack
            stack.append(i)
        
        return max_area
    
# @lc code=end

