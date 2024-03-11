from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import random
import time  


class NQueensCSP:
    """
    Solution of N-Queens problem using Least Constraining value approach and CSP.
    If user enters a path to file, the starting points of quenns are initialized with given instructions.
    Otherwise, the random NxN board is initialized and solved. The output of program is 1 based indexes of queens, 
    time statistics and visualization showing final placement of queens.
    """
    def __init__(self):

        self.domains = {}
        self.constraints = []
            
    def input_reader(self, file_path):
        """
        Funtion which reads input from a file.
        If pass keyword is inputted by user, function proceeds with random configuration.
        """
        
        if file_path=="pass":
            #Generation of random board
            n = int(input("Please enter the size of NxN board: "))
            qcolumns = [random.randint(0, n - 1) for _ in range(n)]
            
        else:
            #reading queen columns from the file
            with open(file_path, 'r') as file:
                qcolumns = [int(line.strip()) for line in file if not line.strip().startswith('#')]
                n = len(qcolumns)
        

        #initializing domains with all possible row values for each column
        self.domains = {i: list(range(n)) for i in range(n)}
        #generating all possible constraints between pairs of variables
        self.constraints = [(i, j) for i in range(n) for j in range(n) if i != j]
        
        
        return qcolumns
    
    def check_consistency(self, current_var, current_val, assignment):
        """
        Function to check if assigning a value to a variable maintains consistency with the current assignment.
        """
        # Checking if the value conflicts with any existing assignments
        for assigned_var, assigned_val in assignment.items():
            if assigned_val == current_val or abs(assigned_val - current_val) == abs(assigned_var - current_var):
                return False
        
        return True


    def select_unassigned(self, assignment):
        """
        Function to select an unassigned variable
        """
        # Selecting the variable with the fewest remaining values in its domain
        unassigned_variables = [variable for variable in range(len(self.domains)) if variable not in assignment]
        result =  min(unassigned_variables, key=lambda variable: (len(self.domains[variable]), -variable))
        return result

    def prioritize_domain_values(self, selected_variable, current_assignment):
        """
        Orders the possible values for a given variable using the Least Constraining Value (LCV) heuristic.
        This method prioritizes values that impose the fewest constraints on neighboring variables.
        """
        def calculate_lcv(value):
            """
            Computes the LCV for a specific value by counting how many options it leaves open for other variables.
            """
            compatibility_count = 0
            for adjacent_variable in range(len(self.domains)):
                # Exclude the current variable and those already assigned
                if adjacent_variable != selected_variable and adjacent_variable not in current_assignment:
                    for possible_value in self.domains[adjacent_variable]:
                        # Check if the value is compatible with the current partial solution
                        if self.check_consistency(adjacent_variable, possible_value, {**current_assignment, selected_variable: value}):
                            compatibility_count += 1
            return compatibility_count

        # Order the values for the selected_variable by their LCV, preferring those with higher counts
        return sorted(self.domains[selected_variable], key=calculate_lcv, reverse=True)

    def revise_domains(self, variable_x, variable_y):
        """
        Updates the domains of two variables to ensure arc consistency is maintained.
        This function iteratively checks and removes values from the domain of variable_x
        that are inconsistent with the domain of variable_y, based on the constraints between them.
        """
        domain_updated = False
        for value_x in self.domains[variable_x][:]:  # Copy of the domain to iterate over
            # Check if there's no value in variable_y's domain compatible with value_x
            if all(not any(value_x == value_y for value_y in self.domains[variable_y]) for value_y in self.domains[variable_y]):
                # If no compatible value found, remove value_x from variable_x's domain
                self.domains[variable_x].remove(value_x)
                domain_updated = True
        return domain_updated

    def apply_arcconsistency_algorithm(self):
        """
        Implements the AC3 algorithm to achieve arc consistency across all variables.
        """
        # Create a queue to hold all the constraints for processing
        constraints_queue = deque(self.constraints)
        while constraints_queue:
            # Remove and process the first constraint from the queue
            current_var, next_var = constraints_queue.popleft()
            # Attempt to enforce arc consistency between current_var and next_var
            if self.revise_domains(current_var, next_var):
                # If the domain of current_var is empty, a solution is not possible
                if not self.domains[current_var]:
                    return False
                # Re-enqueue constraints involving current_var for re-evaluation
                for adjacent_var, _ in self.constraints:
                    if adjacent_var != next_var:
                        constraints_queue.append((adjacent_var, current_var))
        return True


    def attempt_solution(self, current_assignment):
        """
        A recursive utility function to facilitate the backtracking search process.
        """
        # Check if a complete assignment is achieved
        if len(current_assignment) == len(self.domains):
            return current_assignment
        # Identify an unassigned variable for potential assignment
        next_variable = self.select_unassigned(current_assignment)
        # Retrieve a list of viable values for the next_variable based on the LCV heuristic
        viable_values = self.prioritize_domain_values(next_variable, current_assignment)

        for possible_value in viable_values:
            # Ensure the selected value maintains consistency across the board
            if self.check_consistency(next_variable, possible_value, current_assignment):
                # Temporarily assign the selected value to the variable
                current_assignment[next_variable] = possible_value
                # Recursively attempt to build upon the current partial solution
                potential_solution = self.attempt_solution(current_assignment)
                # If a valid solution is formed, return it
                if potential_solution is not None:
                    return potential_solution
                # If the attempt fails, retract the assignment and try the next possibility
                del current_assignment[next_variable]

        # Return None if no valid solution can be constructed
        return None

    
    def visualize_board(self, initial_positions, final_positions):
        # Determine the board size from the domains, assuming a square board
        n = len(self.domains)

        # Create figure with two subplots
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))  

        # Prepare the checkerboard pattern and settings for both plots
        for subplot in ax:
            chessboard = np.zeros((n, n))
            chessboard[1::2, 0::2] = 1
            chessboard[0::2, 1::2] = 1
            subplot.imshow(chessboard, cmap='gray')
            subplot.set_xticks(np.arange(-.5, n, 1), minor=True)
            subplot.set_yticks(np.arange(-.5, n, 1), minor=True)
            subplot.grid(which="minor", color="black", linestyle='-', linewidth=2)
            subplot.tick_params(which="minor", size=0)
            subplot.set_xticks([])
            subplot.set_yticks([])

        # Adjust initial and final positions for 0-based indexing and matplotlib's coordinate system
        # Note: Subtracting 1 from both x and y to align with 0-based indexing
        initial_positions_adj = [(x - 1, y - 1) for x, y in initial_positions]
        final_positions_adj = [(x - 1, y - 1) for x, y in final_positions]

        # Plot initial positions on the left board
        for x, y in initial_positions_adj:
            ax[0].plot(y, x, 'x', color='red', markersize=10)
        ax[0].set_title('Initial Positions')

        # Plot final positions on the right board
        for x, y in final_positions_adj:
            ax[1].plot(y, x, 'o', color='gold', markersize=10)
        ax[1].set_title('Final Positions')

        plt.show()


    def resolve_nqueens(self, input_path):
        """
        Main function which tackles the N-Queens challenge employing Constraint Satisfaction Problem (CSP) strategies.
        """
        start_time = time.time()  # Record the start time

        # Acquire board setup from the provided file path, initializing domains and constraints accordingly
        initial_queen_positions = self.input_reader(input_path)
        initial_queen_positions_forvisual = [(index + 1, number) for index, number in enumerate(initial_queen_positions)]
        # Attempt to ensure arc consistency across all variables
        if self.apply_arcconsistency_algorithm():  # Replaces search_for_solution call with direct AC3 application
            # Initiate a backtracking search to discover a viable solution
            solution_assignment = self.attempt_solution({})
            # Verify and display the outcome
            if solution_assignment is None:
                print("No solution found")
            else:
                print("Solution found")
                queen_final_positions = []
                # Translate the solution into queen positions for visualization
                print()
                for column in range(len(initial_queen_positions)):
                    queen_final_positions.append((column + 1, solution_assignment[column] + 1))
                    print((column + 1, solution_assignment[column] + 1))
        
        end_time = time.time()  # Record the end time
        # Calculate and print the execution time excluding visualization
        execution_time = round((end_time - start_time),3)
        print(f"Execution time (excluding visualization): {execution_time} seconds.")
        
        # Check if a solution was found before calling visualize_board
        if solution_assignment is not None:
            # Utilize the visualization method to display the board configuration
            self.visualize_board(initial_queen_positions_forvisual,queen_final_positions)
        else:
            print("No solution found.")


# Instantiate the NQueensCSP class and invoke the solution method
csp_solver_instance = NQueensCSP()
csp_solver_instance.resolve_nqueens("pass")
#csp_solver_instance.resolve_nqueens(r"path_to_file")
