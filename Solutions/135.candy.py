#
# @lc app=leetcode id=135 lang=python3
#
# [135] Candy
#

# @lc code=start
from typing import List

class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        assigned_candy = [1] * n  # Initialize each child's candy count to 1

        # Left-to-right pass: ensure each child has more candies than the one before if the rating is higher
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                assigned_candy[i] = assigned_candy[i - 1] + 1

        # Right-to-left pass: ensure each child has more candies than the one after if the rating is higher
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                assigned_candy[i] = max(assigned_candy[i], assigned_candy[i + 1] + 1)

        # Sum up the total candies required
        return sum(assigned_candy)
    
# @lc code=end

