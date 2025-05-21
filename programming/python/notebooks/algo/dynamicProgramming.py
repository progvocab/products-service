def longest_common_substring(str1, str2):
    m, n = len(str1), len(str2)
    # Create a 2D DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    max_length = 0
    end_pos = 0  # To track the end position of LCS in str1

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0  # Reset on mismatch

    # Extract the substring
    longest_substr = str1[end_pos - max_length:end_pos]
    return longest_substr, max_length


# Example usage
#s1 = "abcdef"
#s2 = "zcdemf"
#substr, length = #longest_common_substring(s1, s2)
#print("Longest Common Substring:", substr)
#print("Length:", length)
