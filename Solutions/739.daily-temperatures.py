#
# @lc app=leetcode id=739 lang=python3
#
# [739] Daily Temperatures
#

# @lc code=start
from typing import List

class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        if len(temperatures) == 1:
            return [0]  # Single day means no warmer days ahead
        
        stack = []  # Monotonic stack to keep track of indices of decreasing temperatures
        answer = [0] * len(temperatures)  # Result array initialized with 0

        for i in range(len(temperatures)):
            # While the stack is not empty and the current temperature is greater
            # than the temperature at the index stored at the top of the stack
            while stack and temperatures[i] > temperatures[stack[-1]]:
                # Get the index of the last temperature that is smaller
                prev_index = stack.pop()
                # Calculate the number of days until a warmer temperature
                answer[prev_index] = i - prev_index
            
            # Push the current day's index onto the stack
            stack.append(i)

        # Indices left in the stack correspond to days with no warmer temperatures ahead
        # These are already set to 0 in the `answer` array
        
        return answer

# @lc code=end

