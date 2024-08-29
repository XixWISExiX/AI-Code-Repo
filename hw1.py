# Problem: Implement the Breadth-First Search (BFS), Depth-First Search (DFS) 
# and Greedy Best-First Search (GBFS) algorithms on the graph from Figure 1 in hw1.pdf.


# Instructions:
# 1. Represent the graph from Figure 1 in any format (e.g. adjacency matrix, adjacency list).
# 2. Each function should take in the starting node as a string. Assume the search is being performed on
#    the graph from Figure 1.
#    It should return a list of all node labels (strings) that were expanded in the order they where expanded.
#    If there is a tie for which node is expanded next, expand the one that comes first in the alphabet.
# 3. You should only modify the graph representation and the function body below where indicated.
# 4. Do not modify the function signature or provided test cases. You may add helper functions. 
# 5. Upload the completed homework to Gradescope, it must be named 'hw1.py'.

# Examples:
#     The test cases below call each search function on node 'S' and node 'A'
# -----------------------------

                    #A  #B  #C  #D  #E  #F  #G  #H  #I  #J  #K  #L  #M  #N  #P  #Q  #S
adjacencyMatrix = [[-1,  4, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], #A
                   [ 4, -1,  2, -1, -1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], #B
                   [-1,  2, -1, -1, -1, -1, -1,  4, -1, -1, -1, -1, -1, -1, -1, -1,  3], #C
                   [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  8, -1, -1, -1, -1,  2], #D
                   [ 1, -1, -1, -1, -1,  3, -1, -1,  6, -1, -1, -1, -1, -1, -1, -1, -1], #E
                   [-1,  2, -1, -1,  3, -1, -1, -1, -1,  6,  4, -1, -1, -1, -1, -1, -1], #F
                   [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  4,  4, -1, 10, -1], #G
                   [-1, -1,  4, -1, -1, -1, -1, -1, -1, -1,  3,  7, -1, -1, -1, -1, -1], #H
                   [-1, -1, -1, -1,  6, -1, -1, -1, -1,  1, -1, -1,  5, -1, -1, -1, -1], #I
                   [-1, -1, -1, -1, -1,  6, -1, -1,  1, -1,  3, -1, -1,  3, -1, -1, -1], #J
                   [-1, -1, -1, -1, -1,  4, -1,  3, -1,  3, -1,  9, -1, -1,  3, -1, -1], #K
                   [-1, -1, -1,  8, -1, -1, -1,  7, -1, -1,  9, -1, -1, -1, -1, 10, -1], #L
                   [-1, -1, -1, -1, -1, -1,  4, -1,  5, -1, -1, -1, -1, -1, -1, -1, -1], #M
                   [-1, -1, -1, -1, -1, -1,  4, -1, -1,  3, -1, -1, -1, -1,  2, -1, -1], #N
                   [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  3, -1, -1,  2, -1, -1, -1], #P
                   [-1, -1, -1, -1, -1, -1, 10, -1, -1, -1, -1, 10, -1, -1, -1, -1, -1], #Q
                   [-1, -1,  3,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]] #S

letter_to_index = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6,
                   'H':7, 'I':8, 'J':9, 'K':10, 'L':11, 'M':12, 'N':13,
                   'P':14, 'Q':15, 'S':16}

index_to_letter = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G',
                   7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N',
                   14:'P', 15:'Q', 16:'S'}
                
expected_letter_value = {'A':10, 'B':9, 'C':16, 'D':21, 'E':13, 'F':9, 'G':0,
                         'H':12, 'I':9, 'J':5, 'K':8, 'L':18, 'M':3, 'N':4,
                         'P':6, 'Q':9, 'S':17}


def BFS(start: str) -> list:
    # START: Your code here
    visited = []
    queue = [start]
    while queue != []:
        front = queue.pop(0)
        while front in visited: 
            front = queue.pop(0)
        visited.append(front)
        idx = letter_to_index[front]
        for i in range(len(adjacencyMatrix[idx])):
            if adjacencyMatrix[idx][i] != -1:
                letter = index_to_letter[i]
                if letter == 'G':
                    visited.append('G')
                    return visited
                queue.append(letter)
    return visited # Goal state not found
    # END: Your code here


# TODO redo this so you don't have the variables here
def DFS(start: str, visited=[], goal_found=[False]) -> list:
    # START: Your code here
    if start in visited: return visited
    if goal_found[0]: return visited
    visited.append(start)
    if start == 'G':
        goal_found[0] = True
        return visited

    idx = letter_to_index[start]
    for i in range(len(adjacencyMatrix[idx])):
        if adjacencyMatrix[idx][i] != -1:
            letter = index_to_letter[i]
            DFS(letter, visited, goal_found)
    return visited
    # END: Your code here

import heapq # import Priority Queue
def GBFS(start: str) -> list:
    # START: Your code here
    visited = []
    priority_queue = [(expected_letter_value[start], (start))] # sort by the expected value
    while priority_queue != []:
        front = heapq.heappop(priority_queue)[1]
        while front in visited: 
            front = heapq.heappop(priority_queue)[1]
        visited.append(front)
        idx = letter_to_index[front]
        for i in range(len(adjacencyMatrix[idx])):
            value = adjacencyMatrix[idx][i]
            if value != -1:
                letter = index_to_letter[i]
                if letter == 'G':
                    visited.append('G')
                    return visited
                value = expected_letter_value[letter]
                heapq.heappush(priority_queue, (value, letter))
    return visited # Goal state not found
    # END: Your code here



# test cases - DO NOT MODIFY THESE
def run_tests():
    # Test case 1: BFS starting from node 'A'
    assert BFS('A') == ['A', 'B', 'E', 'C', 'F', 'I', 'H', 'S', 'J', 'K', 'M', 'G'], "Test case 1 failed"
    
    # Test case 2: BFS starting from node 'S'
    assert BFS('S') == ['S', 'C', 'D', 'B', 'H', 'L', 'A', 'F', 'K', 'Q', 'G'], "Test case 2 failed"

    # Test case 3: DFS starting from node 'A'
    assert DFS('A', visited=[], goal_found=[False]) == ['A', 'B', 'C', 'H', 'K', 'F', 'E', 'I', 'J', 'N', 'G'], "Test case 3 failed"
    
    # Test case 4: DFS starting from node 'S'
    assert DFS('S', visited=[], goal_found=[False]) == ['S', 'C', 'B', 'A', 'E', 'F', 'J', 'I', 'M', 'G'], "Test case 4 failed"

    # Test case 5: GBFS starting from node 'A'
    assert GBFS('A') == ['A', 'B', 'F', 'J', 'N', 'G'], "Test case 5 failed"
    
    # Test case 6: GBFS starting from node 'S'
    assert GBFS('S') == ['S', 'C', 'B', 'F', 'J', 'N', 'G'], "Test case 6 failed"

    
    
    print("All test cases passed!")

if __name__ == '__main__':
    run_tests()
