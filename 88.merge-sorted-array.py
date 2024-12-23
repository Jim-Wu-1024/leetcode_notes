#
# @lc app=leetcode id=88 lang=python3
#
# [88] Merge Sorted Array
#

# @lc code=start
from typing import List

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Merges two sorted arrays nums1 and nums2 into nums1 in-place.
        """
        # Pointers for nums1, nums2, and the position to insert in nums1
        p1 = m - 1  # Last valid element in nums1
        p2 = n - 1  # Last element in nums2
        p = m + n - 1  # Last position in nums1

        # Merge nums1 and nums2 from the back
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]  # Place nums1[p1] at the current position
                p1 -= 1
            else:
                nums1[p] = nums2[p2]  # Place nums2[p2] at the current position
                p2 -= 1
            p -= 1  # Move the insertion pointer backward

        # Copy remaining elements from nums2, if any
        while p2 >= 0:
            nums1[p] = nums2[p2]
            p2 -= 1
            p -= 1

# @lc code=end

