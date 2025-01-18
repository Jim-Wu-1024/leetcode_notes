### Binary Tree Preorder, Inorder, Postorder Traversal

```python
# ------------------------------- Preorder Traversal --------------------------------------- #
class Solution:
    """
    Recursion-based Solution
    """
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)


class Solution:
    """
    Iteration-based Solution
    """
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        result = []

        stack = [root]
        while stack:
            node = stack.pop()
            result.append(node.val)

            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)

        return result

# ----------------------------------------------------------------------------------------- #
        
# ------------------------------- Inorder Traversal --------------------------------------- #
class Solution:
    """
    Recursion-based Solution
    """
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)


class Solution:
    """
    Iteration-based Solution
    """
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        result = []

        stack = []
        cur = root
        while cur or stack:
            while cur:
                stack.append(cur)
                cur = cur.left

            cur = stack.pop()
            result.append(cur.val)

            cur = cur.right
        
        return result

# ------------------------------------------------------------------------------------------- #

# ------------------------------- Postorder Traversal --------------------------------------- #
class Solution:
    """
    Recursion-based Solution
    """
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]


class Solution:
    """
    Iteration-based Solution
    """
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        result = []
        stack = [root]

        while stack:
            node = stack.pop()
            result.append(node.val)

            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

        return result[::-1]

# ------------------------------------------------------------------------------------------- #

```

### Binary Tree Level Order Traversal

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = deque([root])
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

                level.append(node.val)
            result.append(level)

        return result
        
```

### 1382. Balance a Binary Search Tree

```python
from typing import List, Optional

class Solution:
    def balanceBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        # Step 1: Collect sorted values through in-order traversal
        def in_order(node: Optional[TreeNode]) -> List[int]:
            return in_order(node.left) + [node.val] + in_order(node.right) if node else []

        # Step 2: Build a balanced BST from sorted values
        def build_balanced_tree(start: int, end: int) -> Optional[TreeNode]:
            if start > end:
                return None
            mid = (start + end) // 2
            root = TreeNode(sorted_values[mid])
            root.left = build_balanced_tree(start, mid - 1)
            root.right = build_balanced_tree(mid + 1, end)
            return root

        # Collect sorted values and build the balanced tree
        sorted_values = in_order(root)
        return build_balanced_tree(0, len(sorted_values) - 1)

```

### 1026. Maximum Difference Between Node and Ancestor

```python
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        # Helper function to perform DFS traversal
        # `cur_min` and `cur_max` track the minimum and maximum values along the path
        def dfs(node: Optional[TreeNode], cur_min: int, cur_max: int) -> int:
            if not node:
                # When we reach a leaf node, calculate the difference between max and min
                return cur_max - cur_min

            # Update the current min and max values with the current node's value
            cur_min = min(node.val, cur_min)
            cur_max = max(node.val, cur_max)

            # Recursively calculate the difference in the left and right subtrees
            left_diff = dfs(node.left, cur_min, cur_max)
            right_diff = dfs(node.right, cur_min, cur_max)

            # Return the maximum difference from either subtree
            return max(left_diff, right_diff)

        if not root:
            # If the tree is empty, return 0
            return 0

        # Start DFS from the root with the root's value as both min and max
        return dfs(root, root.val, root.val)

```

### 1372. Longest ZigZag Path in a Binary Tree

```python
class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode], to_left: bool, length: int) -> int:
            if not node:
                return length - 1

            # Explore left and right subtrees based on the current direction
            if to_left:
                # If the direction is to the left, continue left with a new direction (False)
                # and restart the zigzag on the right with length reset to 1
                left = dfs(node.left, False, length + 1)
                right = dfs(node.right, True, 1)
            else:
                # If the direction is to the right, continue right with a new direction (True)
                # and restart the zigzag on the left with length reset to 1
                left = dfs(node.left, False, 1)
                right = dfs(node.right, True, length + 1)

            # Return the maximum path length found in either direction
            return max(left, right)
        
        if not root:
            # Edge case: if the tree is empty, return 0
            return 0

        # Start DFS from the root's left and right subtrees
        return max(dfs(root.left, False, 1), dfs(root.right, True, 1))

```

### 1325. Delete Leaves With a Given Value

```python
class Solution:
    def removeLeafNodes(self, root: Optional[TreeNode], target: int) -> Optional[TreeNode]:
        # Helper function to recursively delete target leaf nodes
        def delete(node: Optional[TreeNode], target: int) -> bool:
            if not node:
                # If the node is None, return False (no leaf to remove)
                return False

            # Recursively check left and right children
            left = delete(node.left, target)
            right = delete(node.right, target)

            # Remove the left child if it was a leaf and its value was equal to the target
            node.left = None if left else node.left

            # Remove the right child if it was a leaf and its value was equal to the target
            node.right = None if right else node.right

            # Return True if the current node is now a leaf and its value equals the target
            return not node.left and not node.right and node.val == target

        if not root:
            # If the tree is empty, return None
            return None

        # Call the helper function on the root
        remove = delete(root, target)

        # If the root itself should be removed, return None; otherwise, return the root
        return None if remove else root

```

### 3319. K-th Largest Perfect Subtree Size in Binary Tree

```python
class Solution:
    """
    Enhancement: Using Heap
    """
    def kthLargestPerfectSubtree(self, root: Optional[TreeNode], k: int) -> int:
        # List to store the sizes of all perfect subtrees
        perfect_sizes = []
        
        def postorder(node) -> int:
            if not node:
                # Base case: if the node is None, return size 0
                return 0

            if not node.left and not node.right:
                # A leaf node is a perfect subtree of size 1
                perfect_sizes.append(1)
                return 1
            
            # Recursively calculate the size of left and right subtrees
            left_size = postorder(node.left)
            right_size = postorder(node.right)
            
            # Check if the current subtree is perfect
            if left_size and right_size and left_size == right_size:
                # If both subtrees are perfect and of the same size, the current subtree is perfect
                subtree_size = 1 + left_size + right_size
                perfect_sizes.append(subtree_size)
                return subtree_size
            else:
                # If the subtree is not perfect, return size 0
                return 0
        
        # Perform postorder traversal starting from the root
        postorder(root)
        
        # Sort the sizes of perfect subtrees in descending order
        perfect_sizes.sort(reverse=True)
        
        # Return the K-th largest size if it exists, otherwise return -1
        return perfect_sizes[k - 1] if k <= len(perfect_sizes) else -1

```

### 2385. Amount of Time for Binary Tree to Be Infected

```python
from collections import deque
from typing import Optional

class Solution:
    def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
        def createGraph(node: Optional[TreeNode]) -> None:
            if not node:
                return
            
            if node.val not in graph:
                graph[node.val] = []
            
            # Connect the current node to its left child
            if node.left:
                graph[node.val].append(node.left.val)
                if node.left.val not in graph:
                    graph[node.left.val] = []
                graph[node.left.val].append(node.val)
            
            # Connect the current node to its right child
            if node.right:
                graph[node.val].append(node.right.val)
                if node.right.val not in graph:
                    graph[node.right.val] = []
                graph[node.right.val].append(node.val)
            
            # Recursively build the graph for the left and right subtrees
            createGraph(node.left)
            createGraph(node.right)

        def bfs(start: int) -> int:
            time = 0  # Tracks the time taken for infection to spread
            visited = set()  # Keep track of visited nodes to prevent revisiting
            queue = deque([start])  # Start BFS from the infected node

            visited.add(start)  # Mark the starting node as visited

            while queue:
                # Process all nodes at the current level
                for _ in range(len(queue)):
                    current = queue.popleft()  # Get the next node in the queue
                    # Add all unvisited neighbors to the queue
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                            visited.add(neighbor)
                # Increment time after processing all nodes at the current level
                if queue:
                    time += 1
            
            return time

        # Step 1: Build the graph representation of the tree
        graph = {}
        createGraph(root)
        
        # Step 2: Perform BFS starting from the given node and calculate time
        return bfs(start)
```

### 988. Smallest String Starting From Leaf

When using the `min` function to compare two strings, it returns the one that is lexicographically smaller

```python
class Solution:
    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
        def dfs(node: Optional[TreeNode], path: str) -> str:
            if not node:
                # Return None for non-existent nodes (base case for null nodes)
                return None

            # Prepend the current character to the path (leaf-to-root order)
            path = chr(ord('a') + node.val) + path

            # If this is a leaf node, return the constructed path
            if not node.left and not node.right:
                return path

            # Recursively get the smallest string from the left and right subtrees
            left_str = dfs(node.left, path)
            right_str = dfs(node.right, path)

            # Compare the strings to find the lexicographically smallest one
            if left_str and right_str:
                return min(left_str, right_str)  # Use min to compare two valid paths

            # If only one path exists, return the valid one
            return left_str if left_str else right_str

        # Start the DFS traversal with an empty path and return the result
        return dfs(root, "")

```


### 2265. Count Nodes Equal to Average of Subtree

```python
class Solution:
    def __init__(self):
        # Initialize a counter to track the number of valid nodes
        self.count = 0

    def averageOfSubtree(self, root: TreeNode) -> int:
        # Helper function to perform DFS and calculate subtree size and sum
        def dfs(node: Optional[TreeNode]) -> Tuple[int, int]:
            if not node:
                # Base case: empty node has size 0 and sum 0
                return 0, 0

            # Recursively compute the size and sum of the left subtree
            left_size, left_sum = dfs(node.left)

            # Recursively compute the size and sum of the right subtree
            right_size, right_sum = dfs(node.right)

            # Calculate the current subtree's size and sum
            cur_sum = left_sum + right_sum + node.val
            cur_size = left_size + right_size + 1

            # Check if the node's value equals the integer average of its subtree
            if cur_sum // cur_size == node.val:
                self.count += 1  # Increment the count if condition is met

            # Return the size and sum of the current subtree
            return cur_size, cur_sum

        # Start the DFS traversal from the root
        dfs(root)

        # Return the final count of valid nodes
        return self.count

```

### 1457. Pseudo-Palindromic Paths in a Binary Tree

```python
class Solution:
    def __init__(self):
        # Initialize the count of pseudo-palindromic paths
        self.count = 0

    def pseudoPalindromicPaths(self, root: Optional[TreeNode]) -> int:
        # Helper function to check if the current path can form a pseudo-palindrome
        def isValid(path: dict) -> bool:
            odd = 0  # Counter for values with odd frequencies
            for value in path.values():
                if value % 2 == 1:  # If the frequency is odd
                    odd += 1
            # A path is pseudo-palindromic if at most one value has an odd frequency
            return odd <= 1

        # Depth-first search to traverse the tree and track node values along the path
        def dfs(node: Optional[TreeNode], path: dict) -> None:
            if not node:  # Base case: if the node is None, return
                return

            # Increment the frequency of the current node's value in the path
            path[node.val] += 1

            # If it's a leaf node, check if the path is pseudo-palindromic
            if not node.left and not node.right:
                if isValid(path):  # Validate the path
                    self.count += 1  # Increment the count if valid
            else:
                # Recursively traverse the left and right subtrees
                dfs(node.left, path)
                dfs(node.right, path)

            # Backtrack: decrement the frequency of the current node's value
            path[node.val] -= 1

        # Dictionary to store the frequency of values (1 to 9) along the current path
        path = {i: 0 for i in range(1, 10)}

        # Start the depth-first search from the root
        dfs(root, path)

        # Return the total count of pseudo-palindromic paths
        return self.count


class Solution:
    def __init__(self):
        self.count = 0

    def pseudoPalindromicPaths(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode], path_mask: int) -> None:
            if not node:
                return
            
            # Toggle the bit corresponding to the current node's value
            path_mask ^= 1 << node.val

            # Check if it's a leaf node
            if not node.left and not node.right:
                # Check if the path is pseudo-palindromic
                if path_mask & (path_mask - 1) == 0:
                    self.count += 1
                return

            dfs(node.left, path_mask)
            dfs(node.right, path_mask)

        dfs(root, 0)
        return self.count

```


### 971. Flip Binary Tree To Match Preorder Traversal

```python
class Solution:
    def __init__(self):
        # Initialize index to track the position in the voyage list
        self.index = 0
        # Initialize result to store the nodes where flips are required
        self.result = []

    def flipMatchVoyage(self, root: Optional[TreeNode], voyage: List[int]) -> List[int]:
        # Define the helper DFS function to traverse the tree
        def dfs(node: Optional[TreeNode]) -> None:
            if not node:  # If the node is None, return (base case for recursion)
                return 

            # If the current node value does not match the voyage at the current index
            if voyage[self.index] != node.val:
                self.result = [-1]  # Mark the result as impossible to match
                return

            # Move to the next index in the voyage
            self.index += 1

            # Check if we need to flip (if the left child does not match the next value in voyage)
            if node.left and voyage[self.index] != node.left.val:
                # Record the current node value as a flip is required
                self.result.append(node.val)
                # Traverse the right subtree first (flipping order)
                dfs(node.right)
                # Traverse the left subtree
                dfs(node.left)
            else:
                # If no flip is needed, traverse the left subtree first
                dfs(node.left)
                # Then traverse the right subtree
                dfs(node.right)

        # Start the DFS traversal from the root
        dfs(root)

        # If the result contains [-1], it means the voyage cannot match the tree structure
        if self.result and self.result[0] == -1:
            return [-1]

        # Return the list of nodes where flips are required
        return self.result

```

### 1145. Binary Tree Coloring Game

```python
class Solution:
    def btreeGameWinningMove(self, root: Optional[TreeNode], n: int, x: int) -> bool:
        def constructGraph(node: Optional[TreeNode]) -> None:
            if not node:
                return

            # Ensure the current node exists in the graph
            if node.val not in graph:
                graph[node.val] = []

            # Add edges for the left child
            if node.left:
                graph[node.val].append(node.left.val)
                graph[node.left.val] = graph.get(node.left.val, [])
                graph[node.left.val].append(node.val)

            # Add edges for the right child
            if node.right:
                graph[node.val].append(node.right.val)
                graph[node.right.val] = graph.get(node.right.val, [])
                graph[node.right.val].append(node.val)

            # Recursively build the graph for the left and right subtrees
            constructGraph(node.left)
            constructGraph(node.right)

        # Function to calculate the size of a connected component using BFS
        def calculateSubtreeSize(start: int, visited: set) -> int:
            size = 0
            queue = deque([start])
            visited.add(start)

            # BFS to count all nodes in the connected component
            while queue:
                current_node = queue.popleft()
                size += 1
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            return size

        # Step 1: Build the graph representation of the binary tree
        graph = {}
        constructGraph(root)

        # Step 2: Calculate the size of each connected component starting from x's neighbors
        component_sizes = []
        for neighbor in graph[x]:
            visited = set()
            visited.add(x)  # Exclude the node x itself
            size = calculateSubtreeSize(neighbor, visited)
            component_sizes.append(size)

        # Step 3: Determine the size of the largest component Player 2 can control
        player1_nodes = sum(component_sizes) - max(component_sizes) + 1 
        largest_player2_nodes = n - player1_nodes

        # Step 4: Player 2 wins if they can control more than half the nodes
        return largest_player2_nodes > player1_nodes

```

### 1315. Sum of Nodes with Even-Valued Grandparent

```python
class Solution:
    def __init__(self):
        # Initialize the total sum of nodes with even-valued grandparents
        self.total = 0

    def sumEvenGrandparent(self, root: Optional[TreeNode]) -> int:
        # Helper function to perform DFS
        def dfs(node: Optional[TreeNode], parent: Optional[TreeNode], grandparent: Optional[TreeNode]) -> None:
            if not node:  # Base case: If the current node is None, return
                return

            # If the grandparent exists and its value is even, add the current node's value to the total sum
            if grandparent and grandparent.val % 2 == 0:
                self.total += node.val

            # Recursively call DFS for the left and right subtrees
            # Update the parent to the current node and the grandparent to the parent
            dfs(node.left, node, parent)
            dfs(node.right, node, parent)

        # Start the DFS from the root node with no parent or grandparent
        dfs(root, None, None)

        # Return the total sum
        return self.total

```