#
# @lc app=leetcode id=738 lang=python3
#
# [738] Monotone Increasing Digits
#

# @lc code=start
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        # Convert the number to a list of characters (digits) for easy manipulation
        digits = list(str(n))

        # Initialize change_point to mark where we start setting digits to '9'
        change_point = len(digits)

        # Traverse the number from the end to the beginning
        for i in range(len(digits) - 1, 0, -1):
            # If the current digit is less than the previous one, we need to adjust
            if digits[i - 1] > digits[i]:
                # Decrement the previous digit by 1
                digits[i - 1] = str(int(digits[i - 1]) - 1)
                # Update the change_point to the current index
                change_point = i

        # Set all digits after change_point to '9' to ensure monotonic increase
        for i in range(change_point, len(digits)):
            digits[i] = '9'

        # Convert the list of characters back to an integer and return it
        return int(''.join(digits))
    
# @lc code=end

