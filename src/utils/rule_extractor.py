"""
Rule extraction utility for Graph Tsetlin Machines.
Converts trained GTM clauses into human-readable rules.
"""

import numpy as np
from typing import List, Dict, Tuple
from GraphTsetlinMachine.graphs import Graphs


class RuleExtractor:
    """Extract and format human-readable rules from trained GTM models."""
    
    def __init__(self, model, symbol_names: List[str]):
        """
        Initialize the rule extractor.
        
        Args:
            model: Trained HexGraphTM model
            symbol_names: List of symbol names (e.g., ['Empty', 'Player0', 'Player1', 'Row0', ...])
        """
        self.model = model
        self.symbol_names = symbol_names
        
    def extract_rules(self, max_rules: int = 100, min_weight: int = 1) -> List[Dict]:
        """
        Extract the most important rules from the trained model.
        
        Args:
            max_rules: Maximum number of rules to extract
            min_weight: Minimum clause weight to include
            
        Returns:
            List of rule dictionaries with 'clause_id', 'weight', 'literals', 'rule_text'
        """
        if self.model.tm is None or not self.model.trained:
            return []
        
        # Get clause weights
        weights = self.model.tm.get_weights()  # Shape: (num_outputs, num_clauses)
        
        # Get clause literals in hypervector format
        clause_literals_hv = self.model.tm.get_hyperliterals(depth=0)
        
        rules = []
        
        # Process each output class
        for class_idx in range(weights.shape[0]):
            class_weights = weights[class_idx]
            
            # Get clauses sorted by absolute weight (importance)
            sorted_indices = np.argsort(np.abs(class_weights))[::-1]
            
            count = 0
            for clause_idx in sorted_indices:
                if count >= max_rules:
                    break
                    
                weight = class_weights[clause_idx]
                
                if abs(weight) < min_weight:
                    continue
                
                # Extract literals for this clause
                literals = clause_literals_hv[clause_idx]
                
                # Convert to human-readable rule
                rule_text = self._format_rule(literals, weight, class_idx)
                
                if rule_text:  # Only add non-empty rules
                    rules.append({
                        'clause_id': clause_idx,
                        'class': class_idx,
                        'weight': int(weight),
                        'literals': literals,
                        'rule_text': rule_text
                    })
                    count += 1
        
        return rules
    
    def _format_rule(self, literals: np.ndarray, weight: float, class_idx: int) -> str:
        """
        Convert a clause's literals into human-readable text.
        
        Args:
            literals: Binary array indicating which literals are included
            weight: Clause weight
            class_idx: Which class this clause predicts
            
        Returns:
            Human-readable rule string
        """
        # Split into positive and negative literals
        num_features = len(literals) // 2
        positive_literals = literals[:num_features]
        negative_literals = literals[num_features:]
        
        conditions = []
        
        # Add positive conditions (symbol must be present)
        for i, active in enumerate(positive_literals):
            if active and i < len(self.symbol_names):
                conditions.append(f"{self.symbol_names[i]}")
        
        # Add negative conditions (symbol must NOT be present)
        for i, active in enumerate(negative_literals):
            if active and i < len(self.symbol_names):
                conditions.append(f"NOT {self.symbol_names[i]}")
        
        if not conditions:
            return ""
        
        # Format the rule
        polarity = "positive" if weight > 0 else "negative"
        class_name = f"Player {class_idx}"
        
        rule = f"IF {' AND '.join(conditions[:10])}"  # Limit to first 10 conditions for readability
        if len(conditions) > 10:
            rule += f" ... ({len(conditions)-10} more)"
        
        rule += f" THEN vote {polarity} for {class_name} (weight: {weight})"
        
        return rule
    
    def save_rules(self, filepath: str, max_rules: int = 100):
        """
        Extract rules and save them to a text file.
        
        Args:
            filepath: Path to save the rules file
            max_rules: Maximum number of rules to extract
        """
        rules = self.extract_rules(max_rules=max_rules)
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GRAPH TSETLIN MACHINE - LEARNED RULES\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total rules extracted: {len(rules)}\n")
            f.write(f"Model: {self.model.params}\n\n")
            
            f.write("="*80 + "\n")
            f.write("RULES (sorted by importance)\n")
            f.write("="*80 + "\n\n")
            
            for i, rule in enumerate(rules, 1):
                f.write(f"Rule #{i} (Clause {rule['clause_id']}, Class {rule['class']}):\n")
                f.write(f"  {rule['rule_text']}\n\n")
        
        print(f"Rules saved to {filepath}")
        print(f"  Total rules: {len(rules)}")
    
    def extract_messages(self, num_edge_types: int = 1) -> Dict:
        """
        Extract message passing information from the trained model.
        
        Args:
            num_edge_types: Number of edge types in the graph (1 for Hex - Adjacent edges)
            
        Returns:
            Dictionary with message information for each depth level
        """
        if self.model.tm is None or not self.model.trained:
            return {}
        
        messages_info = {}
        depth = self.model.params.get('depth', 1)
        num_clauses = self.model.params.get('number_of_clauses', 0)
        
        # Extract messages for each depth level (depth > 1 means message passing)
        if depth > 1:
            for d in range(1, depth):
                try:
                    # Get message literals for this depth
                    # Shape: (edge_types, num_clauses, 2 * num_clauses)
                    messages = self.model.tm.get_messages(depth=d, edge_types=num_edge_types)
                    
                    messages_info[f'depth_{d}'] = {
                        'shape': messages.shape,
                        'messages': messages,
                        'description': f'Messages at depth {d}'
                    }
                except Exception as e:
                    messages_info[f'depth_{d}'] = {
                        'error': str(e),
                        'description': f'Failed to extract messages at depth {d}'
                    }
        
        return messages_info
    
    def save_messages(self, filepath: str, num_edge_types: int = 1):
        """
        Extract and save message passing information to a text file.
        
        Args:
            filepath: Path to save the messages file
            num_edge_types: Number of edge types (1 for Hex)
        """
        messages_info = self.extract_messages(num_edge_types=num_edge_types)
        depth = self.model.params.get('depth', 1)
        num_clauses = self.model.params.get('number_of_clauses', 0)
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GRAPH TSETLIN MACHINE - MESSAGE PASSING ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Model Depth: {depth}\n")
            f.write(f"Number of Clauses: {num_clauses}\n")
            f.write(f"Number of Edge Types: {num_edge_types}\n\n")
            
            if depth <= 1:
                f.write("NOTE: This model has depth=1 (no message passing between nodes).\n")
                f.write("Message passing only occurs when depth > 1.\n")
                f.write("\nTo enable message passing, train with --depth 2 or higher.\n")
            else:
                f.write("="*80 + "\n")
                f.write("MESSAGE PASSING PATTERNS\n")
                f.write("="*80 + "\n\n")
                
                f.write("Message passing allows clauses to aggregate information from neighbors.\n")
                f.write("Each clause can send positive/negative messages to neighboring clauses.\n\n")
                
                for depth_key in sorted(messages_info.keys()):
                    depth_num = depth_key.split('_')[1]
                    info = messages_info[depth_key]
                    
                    f.write(f"\n{'='*80}\n")
                    f.write(f"DEPTH LEVEL {depth_num}\n")
                    f.write(f"{'='*80}\n\n")
                    
                    if 'error' in info:
                        f.write(f"Error: {info['error']}\n")
                        continue
                    
                    messages = info['messages']
                    f.write(f"Shape: {info['shape']}\n")
                    f.write(f"  - Edge types: {messages.shape[0]}\n")
                    f.write(f"  - Clauses: {messages.shape[1]}\n")
                    f.write(f"  - Message bits (2 * num_clauses): {messages.shape[2]}\n\n")
                    
                    # Analyze message patterns for each edge type
                    for edge_type in range(messages.shape[0]):
                        f.write(f"\nEdge Type {edge_type} (Adjacent connections):\n")
                        f.write("-" * 60 + "\n\n")
                        
                        # Show top clauses that send the most messages
                        edge_messages = messages[edge_type]  # Shape: (num_clauses, 2*num_clauses)
                        
                        # Count active messages per clause
                        positive_msgs = edge_messages[:, :num_clauses]  # Positive messages
                        negative_msgs = edge_messages[:, num_clauses:]  # Negative messages
                        
                        clause_activity = np.sum(positive_msgs, axis=1) + np.sum(negative_msgs, axis=1)
                        active_clauses = np.where(clause_activity > 0)[0]
                        
                        f.write(f"Active clauses (sending messages): {len(active_clauses)} / {num_clauses}\n\n")
                        
                        if len(active_clauses) > 0:
                            # Show top 20 most active clauses
                            top_k = min(20, len(active_clauses))
                            sorted_indices = np.argsort(clause_activity)[::-1][:top_k]
                            
                            f.write(f"Top {top_k} most active message-sending clauses:\n\n")
                            
                            for rank, clause_idx in enumerate(sorted_indices, 1):
                                pos_count = int(np.sum(positive_msgs[clause_idx]))
                                neg_count = int(np.sum(negative_msgs[clause_idx]))
                                total = pos_count + neg_count
                                
                                f.write(f"  {rank}. Clause {clause_idx}:\n")
                                f.write(f"      Positive messages: {pos_count}\n")
                                f.write(f"      Negative messages: {neg_count}\n")
                                f.write(f"      Total: {total}\n")
                                
                                # Show which clauses it sends to (top 5)
                                if total > 0:
                                    combined = positive_msgs[clause_idx] + negative_msgs[clause_idx]
                                    target_clauses = np.where(combined > 0)[0]
                                    if len(target_clauses) > 0:
                                        top_targets = target_clauses[:5]
                                        f.write(f"      Sends to clauses: {', '.join(map(str, top_targets))}")
                                        if len(target_clauses) > 5:
                                            f.write(f" ... and {len(target_clauses)-5} more")
                                        f.write("\n")
                                f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("="*80 + "\n\n")
            f.write("Message passing enables clauses to communicate:\n")
            f.write("- Positive messages: Reinforce a clause's decision\n")
            f.write("- Negative messages: Suppress a clause's decision\n")
            f.write("- More active clauses = more information sharing between nodes\n")
            f.write("- Depth determines how far messages propagate through the graph\n\n")
        
        print(f"Message passing analysis saved to {filepath}")
        if depth > 1:
            print(f"  Depth levels analyzed: {list(range(1, depth))}")
        else:
            print("  Note: No message passing (depth=1)")
