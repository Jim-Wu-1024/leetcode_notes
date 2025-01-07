## Hash Table

A Hash Table is a data structure that stores key-value pairs in an efficient manner for fast retrieval, insertion, and deletion operations.

### 242. Valid Anagram

The `==` operator checks if the frequency dictionaries for s and t are identical.

```python
from collections import Counter

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # Compare the two frequency dictionaries
        # If they are equal, s and t are anagrams
        return Counter(s) == Counter(t)

```

### 349. Intersection of Two Arrays

- Union: Combine unique elements from two sets.
   - `set1 | set2` or `set1.union(set2)`
- Intersection: Find common elements.
   - `set1 & set2` or `set1.intersection(set2)`
- Difference: Elements in one set but not the other.
   - `set1 - set2` or `set1.difference(set2)`

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Use set intersection to find common elements
        return list(set(nums1) & set(nums2))

```


### 350. Intersection of Two Arrays II

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Ensure nums1 is smaller for efficiency
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1

        # Step 1: Create a dictionary (dict1) to count occurrences of elements in nums1
        dict1 = {}
        for num in nums1:
            dict1[num] = dict1.get(num, 0) + 1

        # Step 2: Iterate through nums2 and check for matches in dict1
        result = []
        for num in nums2:
            if num in dict1 and dict1[num] > 0:  # If the element exists in dict1 with a positive count
                dict1[num] -= 1                 # Decrease the count in dict1
                result.append(num)              # Add the element to the result list

        # Step 3: Return the result list containing the intersection
        return result

```

### 454. 4Sum II

```python
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        # Step 1: Create a dictionary to store the sums of nums1 and nums2
        dict1 = {}
        for i in range(len(nums1)):
            for j in range(len(nums2)):
                total = nums1[i] + nums2[j]  # Compute the sum
                dict1[total] = dict1.get(total, 0) + 1  # Increment the count in the dictionary

        # Step 2: Count complements using nums3 and nums4
        count = 0
        for k in range(len(nums3)):
            for l in range(len(nums4)):
                complement = -(nums3[k] + nums4[l])  # Compute the complement
                if complement in dict1:  # Check if the complement exists in the dictionary
                    count += dict1[complement]  # Add the frequency to the count

        # Step 3: Return the total count of tuples
        return count

```