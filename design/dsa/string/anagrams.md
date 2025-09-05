On LeetCode, “string canonicalization” isn’t a tag they use directly — but several problems require you to **normalize strings to a canonical form** so they can be compared, grouped, or deduplicated.

These problems typically involve:

* Case folding (`lower()` / `casefold()`)
* Removing punctuation/whitespace
* Sorting or shifting letters
* Normalizing Unicode (less common on LeetCode but still relevant conceptually)

---

## **Common Canonicalization Problem Types on LeetCode**

| Problem ID / Name                                   | Canonicalization Idea                                                     | Key Step                                     |
| --------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------- |
| **49. Group Anagrams**                              | Sort the characters of each word                                          | `"eat"` → `"aet"`                            |
| **242. Valid Anagram**                              | Sort or count characters                                                  | Compare canonical forms                      |
| **819. Most Common Word**                           | Lowercase, remove punctuation                                             | `"Bob!"` → `"bob"`                           |
| **859. Buddy Strings**                              | Not pure canonicalization, but sometimes normalize by sorting in variants |                                              |
| **438. Find All Anagrams in a String**              | Sliding window with canonicalized counts                                  |                                              |
| **266. Palindrome Permutation**                     | Normalize by counting characters                                          |                                              |
| **893. Groups of Special-Equivalent Strings**       | Split into even/odd index chars, sort each                                | Canonical form = `(sorted_even, sorted_odd)` |
| **290. Word Pattern** / **791. Custom Sort String** | Normalize mapping of words to pattern                                     |                                              |
| **953. Verifying an Alien Dictionary**              | Map characters to canonical index in order                                |                                              |

---

## **Example 1 — Group Anagrams (LeetCode 49)**

```python
from collections import defaultdict

def groupAnagrams(strs):
    groups = defaultdict(list)
    for word in strs:
        key = "".join(sorted(word))  # canonical form
        groups[key].append(word)
    return list(groups.values())

print(groupAnagrams(["eat","tea","tan","ate","nat","bat"]))
```

Output:

```
[['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

---

## **Example 2 — Most Common Word (LeetCode 819)**
You are given:

- A paragraph (string of text).

- A list of banned words.

You need to find the most frequent word in the paragraph that is not in the banned list.
Ignore capitalization and punctuation.
```python
import re
from collections import Counter

def mostCommonWord(paragraph, banned):
    banned = set(banned)
    words = re.findall(r'\w+', paragraph.lower())  # lowercase, strip punctuation
    counts = Counter(w for w in words if w not in banned)
    return counts.most_common(1)[0][0]

print(mostCommonWord("Bob hit a ball, the hit BALL flew far after it was hit.", ["hit"]))
```

Output:

```
'ball'
```

---

## **Example 3 — Groups of Special-Equivalent Strings (LeetCode 893)**

You are given an array of strings words.
Two strings are special-equivalent if, after any number of swaps of characters at even indices or swaps of characters at odd indices, the two strings become equal.

Return the number of groups of special-equivalent strings.

```python
def numSpecialEquivGroups(words):
    seen = set()
    for w in words:
        even = sorted(w[::2]) # two step at a time 
        odd = sorted(w[1::2]) 
        seen.add(("".join(even), "".join(odd)))  # canonical form
    return len(seen)

print(numSpecialEquivGroups(["abcd","cdab","adcb","cbad"]))
```

Output:

```
1
```

Explanation: All words are equivalent under allowed swaps.

---

## **Canonicalization Strategy for LeetCode**

1. Identify the *equivalence rule* in the problem.
2. Create a transformation function that converts any valid string to a unique, comparable representation.
3. Store canonical forms in a set, dict, or counter for grouping or comparison.

---

If you want, I can prepare a **“string canonicalization cheat sheet for LeetCode”** with:

* All 15–20 canonicalization-related problems
* Their equivalence rules
* The exact canonical form transformation function for each

Do you want me to prepare that list?
