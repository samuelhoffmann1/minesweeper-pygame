import math
import itertools
from dataclasses import dataclass
from settings import *
from sprites import *

from main import Game

class MineGroup:
    ''' A MineGroup is a set of cells that are known
    to have a certain number of mines.
    For example "cell1, cell2, cell3 have exactly 2 mine" or
    "cell4, cell5 have at least 1 mine".
    This is a basic element for Groups and Subgroups solving methods.
    '''

    def __init__(self, cells, mines, group_type="exactly"):
        ''' Cells in questions. Number of mines that those cells have.
        '''
        # Use set rather than list as it speeds up some calculations
        self.cells = set(cells)
        self.mines = mines

        # Group type ("exactly", "no more than", "at least")
        self.group_type = group_type

        # Calculate hash (for deduplication)
        self.hash = self.calculate_hash()

        # Placeholder for cluster belonging info
        # Mine group is a building block for group clusters.
        # belongs_to_cluster will have the cluster this group belongs to
        self.belongs_to_cluster = None

    def calculate_hash(self):
        ''' Hash of a group. To check if such group is already in self.group
        '''
        # Prepare data for hashing: sort cells, add self.mines
        for_hash = sorted(list(self.cells)) + [self.mines] + [self.group_type]
        # Make immutable
        for_hash = tuple(for_hash)
        # Return hash
        return hash(for_hash)

    def __str__(self):
        ''' List cells, mines and group type
        '''
        out = f"Cell(s) ({len(self.cells)}) "
        out += ",".join([str(cell) for cell in self.cells])
        out += f" have {self.group_type} {self.mines} mines"
        return out


class AllGroups:
    ''' Functions to handle a group of MineGroup object (groups and subgroups):
    deduplicate them, generate subgroups ("at least" and "no more than") etc
    '''

    def __init__(self):
        # Hashes of current groups. For deduplication
        self.hashes = set()
        # List of MineGroups
        self.mine_groups = []

        # self.mine_groups will be holding both groups ("exactly") and
        # subgroups ("at least", "no more than") - it is easier to generate
        # subgroups this way.
        # self.count_groups is a count of regular ("exactly") groups.
        # Used to save time iterating only through groups or subgroups.
        self.count_groups = None

        # Frontier: list of cells that belong to at least ong group
        self.frontier = []

    def reset(self):
        ''' Clear the data of the groups
        '''
        self.hashes = set()
        self.mine_groups = []
        # For some reason result is 0.1% better if I don't reset count_groups
        # It does not make sene to me, I can't find why. But so be it.
        # self.count_groups = None
        self.frontier = []

    def reset_clusters(self):
        ''' Clear "belong to cluster" for each group.
        '''
        for group in self.mine_groups:
            group.belong_to_cluster = None

    def next_non_clustered_groups(self):
        ''' Return a group, that is not a part of any cluster
        '''
        for group in self.mine_groups:
            if group.belong_to_cluster is None and \
               group.group_type == "exactly":
                return group
        return None

    def add_group(self, new_group):
        ''' Check if this group has been added already.
        If not, add to the list.
        '''
        if new_group.hash not in self.hashes:
            self.mine_groups.append(new_group)
            self.hashes.add(new_group.hash)

    def generate_frontier(self):
        ''' Populate self.frontier - list of cells belonging to any group
        '''
        frontier = set()
        for group in self:
            for cell in group.cells:
                frontier.add(cell)
        self.frontier = list(frontier)

    def __iter__(self):
        ''' Iterator for the groups
        '''
        return iter(self.mine_groups)

    def exact_groups(self):
        ''' Iterator, but only "exactly" groups
        '''
        return itertools.islice(self.mine_groups, self.count_groups)

    def __str__(self):
        ''' Some info about the groups (for debugging)
        '''
        return f"MineGroups contains {len(self.mine_groups)} groups"


# This method used for solving clusters and for brute force probabilities,
# So it is outside of classes
def all_mines_positions(cells_count, mines_to_set):
    ''' Generate all permutations for "mines_to_set" mines to be in
    "cells_count" cells.
    Result is a list of tuples like (False, False, True, False),
    indicating if the item was chosen (if there is a mine in the cell).
    For example, for generate_mines_permutations(2, 1) the output is:
    [(True, False), (False, True)]
    '''

    def recursive_choose_generator(current_combination, mines_to_set):
        ''' Recursive part of "Choose without replacement" permutation
        generator, results are put into outside "result" variable
        '''
        # No mines to set: save results, out of the recursion
        if mines_to_set == 0:
            result.add(tuple(current_combination))
            return

        for position, item in enumerate(current_combination):
            # Find all the "False" (not a mine) cells
            if not item:
                # Put a mine in it, go to the next recursion level
                current_copy = current_combination.copy()
                current_copy[position] = True
                recursive_choose_generator(current_copy, mines_to_set - 1)

    result = set()
    all_cells_false = [False for _ in range(cells_count)]
    recursive_choose_generator(all_cells_false, mines_to_set)
    return result


class GroupCluster:
    ''' GroupCluster are several MineGroups connected together. All groups
    overlap at least with one other groups o  mine/safe in any of the cell
    can potentially trigger safe/mine in any other cell of the cluster.
    Is a basic element for CSP method
    '''

    def __init__(self, group=None):
        # Cells in the cluster (as a set)
        self.cells_set = set()
        # List if groups in the cluster
        self.groups = []

        # Initiate by adding the first group
        if group is not None:
            self.add_group(group)

        # Placeholder for a list of cells (same as self.cells_set, but list).
        # We use both set and list because set is better for finding overlap
        # and list is needed to solve cluster.
        self.cells = []

        # Placeholder for the solutions of this CSP
        # (all valid sets of mines and safe cells)
        # Positions corresponds to self.cells
        self.solutions = []

        # Placeholder for the resulting frequencies of mines in each cell
        self.frequencies = {}

        # Placeholder for solution weight - how probable is this solution
        # based on the number of mines in it
        self.solution_weights = []

        # Dict of possible mine counts {mines: mine_count, ...}
        self.probable_mines = {}

        # Dict that holds information about "next safe" cells. How many
        # guaranteed safe cells will be in the cluster if "cell" is
        # clicked (and discovered to be safe)
        self.next_safe = {}

    def add_group(self, group):
        ''' Adding group to a cluster (assume they overlap).
        Add group's cells to the set of cells
        Also, mark the group as belonging to this cluster
        '''
        # Total list of cells in the cluster (union of all groups)
        self.cells_set = self.cells_set.union(group.cells)
        # List of groups belonging to the cluster
        self.groups.append(group)
        # Mark the group that it has been used
        group.belong_to_cluster = self

    def overlap(self, group):
        ''' Check if cells in group overlap with cells in the cluster.
        '''
        return len(self.cells_set & group.cells) > 0

    def solve_cluster(self, remaining_mines):
        ''' Use CSP to find the solution to the CSP. Solution is the list of
        all possible mine/safe variations that fits all groups' condition.
        Solution is in the form of a list of Tru/False (Tru for mine,
        False for safe), with positions in the solution corresponding to cells
        in self.cells list. Solution will be stored in self.solutions
        Will result in empty solution if the initial cluster is too big.
        '''

        # We need to fix the order of cells, for that we populate self.cells
        self.cells = list(self.cells_set)

        # We also need a way to find a position of each cell by
        # the cell itself. So here's the addressing dict
        cells_positions = {cell: pos for pos, cell in enumerate(self.cells)}

        # The list to put all the solutions in.
        # Each "solution" is a list of [True, False, None, ...],
        # corresponding to cluster's ordered cells,
        # where True is a mine, False is safe and None is unknown
        # It starts with all None and will be updated for each group
        solutions = [[None for _ in range(len(self.cells))], ]

        for group in self.groups:

            # List of all possible ways mines can be placed in
            # group's cells: for example: [(False, True, False), ...]
            mine_positions = all_mines_positions(len(group.cells),
                                                 group.mines)
            # Now the same, but with cells as keys
            # For example: {cell1: False, cell2: True, cell3: False}
            mine_positions_dict = [dict(zip(group.cells, mine_position))
                                   for mine_position in mine_positions]

            # This is where will will put solutions,
            # updated with current group's conditions
            updated_solutions = []

            # Go through each current solution and possible
            # mine distribution in a group
            for solution in solutions:
                for mine_position in mine_positions_dict:

                    updated_solution = solution.copy()

                    # Check if this permutation fits with this solution
                    for cell in group.cells:
                        # This is the position of this cell in the solution
                        position = cells_positions[cell]
                        # If solution has nothing about this cell,
                        # just update it with cell data
                        if updated_solution[position] is None:
                            updated_solution[position] = mine_position[cell]
                        # But if there is already mine or safe in the solution:
                        # it should be the same as in the permutation
                        # If it isn't: break to the next permutation
                        elif updated_solution[position] != mine_position[cell]:
                            break
                    # If we didn't break (solution and permutation fits),
                    # Add it to the next round
                    else:
                        updated_solutions.append(updated_solution)

            solutions = updated_solutions

        # Check if there are no more mines in solutions than remaining mines
        for solution in solutions:
            mine_count = sum(1 for mine in solution if mine)
            if mine_count <= remaining_mines:
                self.solutions.append(solution)

    def calculate_frequencies(self):
        ''' Once the solution is there, we can calculate frequencies:
        how often is a cell a mine in all solutions. Populates the
        self.frequencies with a dict {cell: frequency, ... },
        where frequency ranges from 0 (100% safe) to 1 (100% mine).
        Also, use weights  (self.solution_weights) - it shows in how many
        cases this solution is likely to appear.
        '''
        # Can't do anything if there are no solutions
        if not self.solutions:
            return

        for position, cell in enumerate(self.cells):
            count_mines = 0
            for solution_n, solution in enumerate(self.solutions):
                # Mine count takes into account the weight of the solution
                # So if fact it is 1 * weight
                if solution[position]:
                    count_mines += self.solution_weights[solution_n]
            # Total in this case - not the number of solutions,
            # but weight of all solutions
            if sum(self.solution_weights) > 0:
                self.frequencies[cell] = count_mines / \
                    sum(self.solution_weights)

    def calculate_next_safe(self):
        ''' Populate self.next_safe, how many guaranteed safe cells
        will be there next move, after clicking this cell.
        '''
        # Result will be here as {cell: next_safe_count}
        next_safe = {}

        # Go through all cells, we'll calculate next_safe for each
        for next_cell_position, next_cell in enumerate(self.cells):
            # Counter of next_safe
            next_safe_counter = 0

            # Go through all cells
            for position, _ in enumerate(self.cells):
                # Ignore itself (the cell we are counting the next_safe_for)
                if position == next_cell_position:
                    continue
                # Now look at all solutions
                for solution in self.solutions:
                    # Skip the solutions where next_cell is a mine
                    if solution[next_cell_position]:
                        continue
                    # If any solution has a mine in "position" -
                    # it will not be safe in the next move
                    if solution[position]:
                        break
                # But if you went through all solutions and all are safe:
                # this is the a cell that will be safe next move
                else:
                    next_safe_counter += 1
            # After looking at all positions we have the final count
            next_safe[next_cell] = next_safe_counter

        self.next_safe = next_safe

    def safe_cells(self):
        ''' Return list of guaranteed safe cells (0 in self.frequencies)
        '''
        safe = [cell for cell, freq in self.frequencies.items() if freq == 0]
        return safe

    def mine_cells(self):
        '''Return list of guaranteed mines (1 in self.frequencies)
        '''
        mines = [cell for cell, freq in self.frequencies.items() if freq == 1]
        return mines

    def calculate_solution_weights(self, covered_cells, remaining_mines):
        ''' Calculate how probable each solution  is,
        if there are total_mines left in the field. That is,
        how many combinations of mines are possible with this solution.
        Populate self.solution_weights with the results
        '''
        self.solution_weights = []

        # For each solution we calculate how  many combination are possible
        # with the remaining mines on the remaining cells
        for solution in self.solutions:
            solution_mines = sum(1 for mine in solution if mine)
            solution_comb = math.comb(len(covered_cells) - len(solution),
                                      remaining_mines - solution_mines)
            self.solution_weights.append(solution_comb)

    def possible_mine_counts(self):
        ''' Based on solution and weights, calculate a dict with possible
        mine counts. For example, {3: 4, 2: 1},  4 solutions with 3 mines,
        1 solution with 2. Put it in self.probable_mines
        Will be used for CSP Leftovers probability calculation.
        '''

        # Cluster was solved
        if self.solutions:
            # Look at each solution
            for solution in self.solutions:
                mines_count = sum(1 for position in solution if position)
                if mines_count not in self.probable_mines:
                    self.probable_mines[mines_count] = 0
                self.probable_mines[mines_count] += 1
            return

    def __str__(self):
        output = f"Cluster with {len(self.groups)} group(s) "
        output += f"and {len(self.cells_set)} cell(s): {self.cells_set}"
        return output


class AllClusters:
    ''' Class that holds all clusters and leftovers data
    '''

    def __init__(self, covered_cells, remaining_mines, game):
        # List of all clusters
        self.clusters = []

        # Cells that are not in any cluster
        self.leftover_cells = set()

        # Mine count and chances in leftover cells
        self.leftover_mines_chances = {}

        # Average chance of a mine in leftover cells (None if NA)
        self.leftover_mine_chance = None

        # Bring these two from the solver object
        self.covered_cells = covered_cells
        self.remaining_mines = remaining_mines

        # And a helper from solver class
        self.game = game

    def calculate_all(self):
        '''Perform all cluster-related calculations: solve clusters,
        calculate weights and mine frequencies etc.
        '''
        for cluster in self.clusters:

            # Solve the cluster, including weights,
            # frequencies and probable mines
            cluster.solve_cluster(self.remaining_mines)
            cluster.calculate_solution_weights(self.covered_cells,
                                               self.remaining_mines)
            cluster.calculate_frequencies()
            cluster.possible_mine_counts()

    def calculate_all_next_safe(self):
        ''' Calculate and populate "next_safe" information for each cluster
        '''
        for cluster in self.clusters:
            cluster.calculate_next_safe()

@dataclass
class CellProbability:
    ''' Data about mine probability for one cell
    '''
    # Which cell we are talking about (coordinates)
    cell: tuple

    # Which method was used to generate mine chance (for statistics)
    source: str

    # Chance this cell is a mine (0 to 1)
    mine_chance: float

    # Chance it would be an opening (no mines in surrounding cells)
    opening_chance: float = 0

    # Frontier value: how many groups is this cell a part of
    frontier: int = 0

    # How many guaranteed safe cells are there for the next move, if this
    # cell is clicked (based on CSP clusters)
    csp_next_safe: int = 0

    # Chance of having at least 1 safe (or mine) cell if this cell is clicked)
    next_move_safe_chance: float = 0

    # Expected number of safe cells for the next move
    next_safe_count: float = 0

    # Survival chance after this move and the next (calculated by going
    # through solving the field with this cell opened)
    next_survival: float = 0

    # Has this mine been shortlisted (selected for the 2nd move survival test)
    shortlisted: int = 0


class AllProbabilities():
    '''Class to work with probability-based information about cells
    Based on list (a list of CellProbability objects)
    '''

    def __init__(self):
        # Dict of AllProbabilities
        self.cells = {}
        # Same, but list (it is easier to collect data into the dict,
        # buts I will need to sort it, so I will convert it into the list)
        self.cells_list = []

    def get_luckiest(self):
        ''' Using information about mine probability ,opening probability
        and so on, return a list of cells with the best chances.
        Also, if next_moves > 0 look into future games and use chances of mine
        in the future moves. For that, use "original_solver" - it has current
        field information
        deterministic: should the solver for the next move be deterministic
        '''

        def simple_best_probability():
            ''' Calculating probability without looking into next moves:
            just pick lowest mine chance with highest opening chance.
            Return a list if several.
            '''
            # The self array of probabilities is sorted, so so the 0's
            # element has the best set of all parameters

            best_mine_chance = self.cells_list[0].mine_chance
            best_opening_chance = self.cells_list[0].opening_chance
            best_frontier = self.cells_list[0].frontier
            best_csp_next_safe = self.cells_list[0].csp_next_safe

            # Pick cells with parameters as good as the best one
            best_cells = []
            for probability_info in self.cells_list:
                if probability_info.mine_chance == best_mine_chance and \
                   probability_info.opening_chance == best_opening_chance and \
                   probability_info.frontier == best_frontier and \
                   probability_info.csp_next_safe == best_csp_next_safe:
                    best_cells.append(probability_info.cell)

            return best_cells

        # Convert dict into list, so we could sort it
        self.cells_list = list(self.cells.values())

        # Sort the cells by: chance, opening, frontier, next_safe
        self.cells_list.sort(key=lambda x: (-x.mine_chance, x.opening_chance,
                                            x.frontier, x.csp_next_safe),
                             reverse=True)

        # Simple best cells are those we can calculate without looking into
        # next move. Return them if we don't need to look into next moves
        simple_best_cells = simple_best_probability()

        return simple_best_cells