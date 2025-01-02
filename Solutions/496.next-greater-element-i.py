#
# @lc app=leetcode id=496 lang=python3
#
# [496] Next Greater Element I
#

# @lc code=start
from typing import List

class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Initialize the answer array with -1 (default when no next greater element exists)
        ans = [-1] * len(nums1)

        # Create a mapping of elements in nums1 to their indices
        nums_12_map = {num: i for i, num in enumerate(nums1)}

        # Monotonic stack to find the next greater element
        stack = []

        # Traverse nums2 to find next greater elements
        for j in range(len(nums2)):
            # While the stack is not empty and the current number is greater than
            # the number corresponding to the index at the top of the stack
            while stack and nums2[j] > nums2[stack[-1]]:
                # Check if the number at the top of the stack exists in nums1
                if nums2[stack[-1]] in nums_12_map:
                    # Update the answer for the corresponding index in nums1
                    ans[nums_12_map[nums2[stack[-1]]]] = nums2[j]
                stack.pop()  # Pop the index from the stack

            # Push the current index onto the stack
            stack.append(j)

        # Return the answer array with next greater elements for nums1
        return ans
    
# @lc code=end

