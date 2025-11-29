"""
Dataset builder for converting Hex game data to Graph Tsetlin Machine format.
Creates Graphs objects with proper node features and edges for GTM training.
"""

import numpy as np
from typing import Dict, Tuple, List
from GraphTsetlinMachine.graphs import Graphs


class DatasetBuilder:
    """
    Convert Hex game data to Graph Tsetlin Machine format.
    """

    def __init__(self, board_size: int = 10):
        """
        Initialize the dataset builder.

        Args:
            board_size: Size of the Hex board
        """
        self.board_size = board_size
        self.num_nodes = board_size * board_size

        # Define symbols for node properties
        # BINARY ENCODING: Player1 inferred via (NOT-Empty AND NOT-Player0)
        # This leverages TM's negation strength and reduces feature space
        # CRITICAL: Add position symbols so GTM knows which edge nodes are on!
        position_symbols = []
        for i in range(board_size):
            position_symbols.append(f'Row{i}')
            position_symbols.append(f'Col{i}')
        
        # REMOVED: 'Player1' symbol (inferred via negation)
        self.symbols = ['Empty', 'Player0'] + position_symbols

        # Pre-compute neighbor structure for hexagonal grid
        self._neighbor_map = self._build_neighbor_map()

    def _build_neighbor_map(self) -> Dict[str, List[str]]:
        """
        Build a map of neighbors for each node in the hexagonal grid.
        Node names are formatted as 'R{row}C{col}' (e.g., 'R0C0', 'R0C1', etc.)

        Returns:
            Dictionary mapping node names to lists of neighbor names
        """
        neighbor_map = {}

        # Hexagonal neighbor offsets
        hex_offsets = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        for row in range(self.board_size):
            for col in range(self.board_size):
                node_name = f'R{row}C{col}'
                neighbors = []

                for dr, dc in hex_offsets:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                        neighbor_name = f'R{nr}C{nc}'
                        neighbors.append(neighbor_name)

                neighbor_map[node_name] = neighbors

        return neighbor_map

    def _board_to_node_name(self, row: int, col: int) -> str:
        """Convert board coordinates to node name."""
        return f'R{row}C{col}'

    def _get_all_node_names(self) -> List[str]:
        """Get all node names in order."""
        names = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                names.append(f'R{row}C{col}')
        return names

    def create_graphs_from_game_data(
        self,
        board_states: np.ndarray,
        winners: np.ndarray,
        hypervector_size: int = 256,
        hypervector_bits: int = 4,
        verbose: bool = True
    ) -> Tuple[Graphs, np.ndarray]:
        """
        Convert game board states to GTM Graphs format.

        Args:
            board_states: Array of shape (num_games, board_size, board_size)
                         with values 0 (empty), 1 (player 0), 2 (player 1)
            winners: Array of shape (num_games,) with winner labels (0 or 1)
            hypervector_size: Size of hypervectors for symbol encoding
            hypervector_bits: Number of bits per symbol
            verbose: Whether to print progress

        Returns:
            (graphs, labels) tuple where:
                - graphs: Graphs object ready for GTM training
                - labels: NumPy array of class labels
        """
        num_games = len(board_states)

        if verbose:
            print(f"\nBuilding GTM Graphs object for {num_games} games...")
            print(f"Board size: {self.board_size}x{self.board_size} ({self.num_nodes} nodes per graph)")
            print(f"Symbols: {self.symbols}")
            print(f"Hypervector: size={hypervector_size}, bits={hypervector_bits}")

        # Initialize Graphs object
        graphs = Graphs(
            num_games,
            symbols=self.symbols,
            hypervector_size=hypervector_size,
            hypervector_bits=hypervector_bits
        )

        # Step 1: Set number of nodes for each graph
        if verbose:
            print("\nStep 1/4: Setting number of nodes...")

        for graph_id in range(num_games):
            graphs.set_number_of_graph_nodes(graph_id, self.num_nodes)

        # Step 2: Prepare node configuration
        if verbose:
            print("Step 2/4: Preparing node configuration...")

        graphs.prepare_node_configuration()

        # Step 3: Add nodes with edges
        if verbose:
            print("Step 3/4: Adding nodes and edges...")

        # Add all nodes first
        for graph_id in range(num_games):
            for row in range(self.board_size):
                for col in range(self.board_size):
                    node_name = self._board_to_node_name(row, col)
                    # Each node has outgoing edges to its neighbors
                    num_edges = len(self._neighbor_map[node_name])
                    graphs.add_graph_node(graph_id, node_name, num_edges)

        # Prepare edge configuration
        graphs.prepare_edge_configuration()

        # Add edges between neighbors
        edge_type = "Adjacent"  # Single edge type for hex adjacency

        for graph_id in range(num_games):
            for node_name, neighbors in self._neighbor_map.items():
                for neighbor_name in neighbors:
                    graphs.add_graph_node_edge(
                        graph_id,
                        node_name,
                        neighbor_name,
                        edge_type
                    )

        # Step 4: Add node properties based on board state
        if verbose:
            print("Step 4/4: Adding node properties...")

        for graph_id in range(num_games):
            board = board_states[graph_id]

            for row in range(self.board_size):
                for col in range(self.board_size):
                    node_name = self._board_to_node_name(row, col)
                    cell_value = board[row, col]

                    # Map cell value to symbol (BINARY ENCODING)
                    if cell_value == 0:
                        symbol = 'Empty'
                        graphs.add_graph_node_property(graph_id, node_name, symbol)
                    elif cell_value == 1:
                        symbol = 'Player0'
                        graphs.add_graph_node_property(graph_id, node_name, symbol)
                    elif cell_value == 2:
                        # Player1 (Blue): NO piece property added!
                        # GTM will infer via negation: NOT-Empty AND NOT-Player0
                        # This forces optimal use of TM's negation capabilities
                        pass  # Skip property assignment for Player1
                    else:
                        raise ValueError(f"Invalid cell value: {cell_value}")
                    
                    # CRITICAL FIX: Add position information!
                    # This tells GTM which edge the node is on
                    graphs.add_graph_node_property(graph_id, node_name, f'Row{row}')
                    graphs.add_graph_node_property(graph_id, node_name, f'Col{col}')

        #  Encode the graphs to finalize the representation
        if verbose:
            print("\nStep 5/5: Encoding graphs...")

        graphs.encode()

        # Convert winners to labels (already 0 or 1)
        labels = winners.astype(np.uint32)

        if verbose:
            print("\nGraph construction complete!")
            print(f"Total graphs: {num_games}")
            print(f"Nodes per graph: {self.num_nodes}")
            print(f"Class distribution: Player 0: {np.sum(labels == 0)}, Player 1: {np.sum(labels == 1)}")

        return graphs, labels

    def create_multiple_datasets(
        self,
        game_data: Dict,
        hypervector_size: int = 256,
        hypervector_bits: int = 4,
        verbose: bool = True
    ) -> Dict[str, Tuple[Graphs, np.ndarray]]:
        """
        Create multiple GTM datasets from game data at different stages.

        Args:
            game_data: Dictionary from GameGenerator with keys like 'states_at_end', 'states_at_-2', etc.
            hypervector_size: Size of hypervectors
            hypervector_bits: Number of bits per symbol
            verbose: Whether to print progress

        Returns:
            Dictionary mapping stage names to (graphs, labels) tuples
        """
        results = {}

        # Find all state keys
        state_keys = [key for key in game_data.keys() if key.startswith('states_at_')]

        winners = game_data['winners']

        for state_key in state_keys:
            stage_name = state_key.replace('states_at_', '')

            if verbose:
                print(f"\n{'='*60}")
                print(f"Creating dataset for stage: {stage_name}")
                print(f"{'='*60}")

            board_states = game_data[state_key]
            graphs, labels = self.create_graphs_from_game_data(
                board_states,
                winners,
                hypervector_size=hypervector_size,
                hypervector_bits=hypervector_bits,
                verbose=verbose
            )

            results[stage_name] = (graphs, labels)

        return results


if __name__ == "__main__":
    # Test the dataset builder
    print("Testing DatasetBuilder...")

    # Create some dummy game data
    board_size = 5
    num_games = 10

    # Random board states
    board_states = np.random.randint(0, 3, size=(num_games, board_size, board_size))
    winners = np.random.randint(0, 2, size=num_games)

    # Build dataset
    builder = DatasetBuilder(board_size=board_size)
    graphs, labels = builder.create_graphs_from_game_data(
        board_states,
        winners,
        hypervector_size=64,
        hypervector_bits=2,
        verbose=True
    )

    print("\nDataset created successfully!")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")
