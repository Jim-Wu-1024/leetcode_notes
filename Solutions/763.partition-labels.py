#
# @lc app=leetcode id=763 lang=python3
#
# [763] Partition Labels
#

# @lc code=start
from typing import List

class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last_occurrence = {char: idx for idx, char in enumerate(s)}

        result = []
        left, right = 0, 0
        for i, char in enumerate(s):
            right = max(right, last_occurrence[char])

            if i == right:
                result.append(right - left + 1)
                left = right + 1

        return result
        
# @lc code=end

