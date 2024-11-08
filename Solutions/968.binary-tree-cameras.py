#
# @lc app=leetcode id=968 lang=python3
#
# [968] Binary Tree Cameras
#

# @lc code=start
from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        # Define state constants to represent each node's coverage status
        COVERED = 0         # Node is covered but does not have a camera
        HAS_CAMERA = 1      # Node has a camera
        NOT_COVERED = 2     # Node is not covered by any camera

        result = 0  # Initialize camera count to zero

        # Define a helper function to perform post-order traversal on the tree
        def traversal(cur: Optional[TreeNode]) -> int:
            nonlocal result
            # If node is None (base case for leaf children), it is covered by default
            if not cur:
                return COVERED

            # Recursively check the left and right children states
            left_state = traversal(cur.left)
            right_state = traversal(cur.right)

            # Case 1: If both children are COVERED, the current node is NOT_COVERED
            # It will rely on its parent to cover it.
            if left_state == COVERED and right_state == COVERED:
                return NOT_COVERED

            # Case 2: If either child is NOT_COVERED, place a camera at the current node
            # This covers both the node and its children.
            if left_state == NOT_COVERED or right_state == NOT_COVERED:
                result += 1
                return HAS_CAMERA

            # Case 3: If either child has a camera, the current node is covered
            # No need for a camera here.
            if left_state == HAS_CAMERA or right_state == HAS_CAMERA:
                return COVERED

            # This line should not be reached with correct input and logic
            return -1

        # After traversal, check if the root node is covered
        # If root is still NOT_COVERED, add one final camera at the root
        if traversal(root) == NOT_COVERED:
            result += 1

        # Return the total number of cameras needed
        return result
# @lc code=end

