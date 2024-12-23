#
# @lc app=leetcode id=42 lang=python3
#
# [42] Trapping Rain Water
#

# @lc code=start
from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        if len(height) < 3:  # Less than 3 bars cannot trap any water
            return 0
        
        stack = []  # Monotonic stack to store indices of the bars
        volume = 0  # Total water trapped

        # Traverse through the heights
        for i in range(len(height)):
            # While the current height is greater than the height of the bar at the top of the stack
            while stack and height[i] > height[stack[-1]]:
                bottom = stack.pop()  # The bar at the top of the stack serves as the bottom of the trapped water
                
                if not stack:
                    break  # No left boundary for the water

                # Calculate water trapped above the current bottom bar
                left = height[stack[-1]]  # Height of the left boundary
                right = height[i]  # Height of the right boundary
                h = min(left, right) - height[bottom]  # Effective height of trapped water
                w = i - stack[-1] - 1  # Width between the left and right boundaries
                volume += h * w  # Accumulate the water volume

            # If the current height is the same as the height at the stack's top, pop it (optional)
            if stack and height[i] == height[stack[-1]]:
                stack.pop()

            stack.append(i)  # Push the current index onto the stack

        return volume

# @lc code=end

