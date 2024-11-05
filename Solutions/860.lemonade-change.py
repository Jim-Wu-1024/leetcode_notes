#
# @lc app=leetcode id=860 lang=python3
#
# [860] Lemonade Change
#

# @lc code=start
from typing import List

class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five_count, ten_count = 0, 0  # Counters for $5 and $10 bills

        for bill in bills:
            if bill == 5:
                # Accept $5 bill, no change needed
                five_count += 1

            elif bill == 10:
                # Accept $10 bill, need to give back one $5 bill as change
                if five_count >= 1:
                    five_count -= 1
                    ten_count += 1
                else:
                    return False  # Not enough $5 bills to give change

            elif bill == 20:
                # Accept $20 bill, prefer to give back one $10 and one $5 if possible
                if ten_count >= 1 and five_count >= 1:
                    ten_count -= 1
                    five_count -= 1
                elif five_count >= 3:
                    # Otherwise, give three $5 bills as change
                    five_count -= 3
                else:
                    return False  # Not enough bills to give change

        # If we never ran out of change, return True
        return True
    
# @lc code=end

