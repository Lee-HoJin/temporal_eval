"""Metadata handling for synthetic data evaluation without external dependencies.

This module provides a lightweight Metadata class for managing database schema
without requiring SDV or other external packages.
"""

import json
import os
from typing import Union, Optional
from pathlib import Path


class Metadata:
    """Lightweight metadata class for managing database schema.

    This class manages metadata for relational databases including tables,
    columns, primary keys, foreign keys, and relationships.

    Parameters
    ----------
    dataset_name: str, default=""
        Name of the dataset.
    """

    def __init__(self, dataset_name: str = ""):
        """Initialize the Metadata object.

        Parameters
        ----------
        dataset_name: str, default=""
            Name of the dataset.
        """
        self.dataset_name = dataset_name
        self.tables = {}
        self.relationships = []

    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]) -> "Metadata":
        """Load metadata from a JSON file.

        Parameters
        ----------
        filepath: Union[str, Path]
            Path to the JSON metadata file.

        Returns
        -------
        Metadata
            Loaded metadata object.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metadata = cls()
        
        # Load tables
        if 'tables' in data:
            metadata.tables = data['tables']
        
        # Load relationships
        if 'relationships' in data:
            metadata.relationships = data['relationships']
        
        return metadata

    def save_to_json(self, filepath: Union[str, Path]):
        """Save metadata to a JSON file.

        Parameters
        ----------
        filepath: Union[str, Path]
            Path to save the JSON metadata file.
        """
        data = {
            'tables': self.tables,
            'relationships': self.relationships
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_tables(self) -> list:
        """Get a list of all table names in the metadata.

        Returns
        -------
        list
            List of table names.
        """
        return list(self.tables.keys())

    def get_primary_key(self, table_name: str) -> Optional[str]:
        """Get the primary key of a table.

        Parameters
        ----------
        table_name: str
            Name of the table.

        Returns
        -------
        Optional[str]
            Name of the primary key column, or None if not found.
        """
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found in metadata")
        
        return self.tables[table_name].get('primary_key')

    def get_table_meta(self, table_name: str) -> dict:
        """Get metadata for a specific table.

        Parameters
        ----------
        table_name: str
            Name of the table.

        Returns
        -------
        dict
            Table metadata as a dictionary.
        """
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found in metadata")
        
        return self.tables[table_name]

    def get_children(self, table_name: str) -> set:
        """Get all child tables of a given table.

        Parameters
        ----------
        table_name: str
            Name of the parent table.

        Returns
        -------
        set
            Set of child table names.
        """
        children = set()
        for relation in self.relationships:
            if relation.get("parent_table_name") == table_name:
                children.add(relation.get("child_table_name"))
        return children

    def get_parents(self, table_name: str) -> set:
        """Get all parent tables of a given table.

        Parameters
        ----------
        table_name: str
            Name of the child table.

        Returns
        -------
        set
            Set of parent table names.
        """
        parents = set()
        for relation in self.relationships:
            if relation.get("child_table_name") == table_name:
                parents.add(relation.get("parent_table_name"))
        return parents
    
    def get_relationships(self, table_name: str) -> list:
        """Get all relationships involving a specific table.

        Parameters
        ----------
        table_name: str
            Name of the table.

        Returns
        -------
        list
            List of relationship dictionaries.
        """
        rels = []
        for relation in self.relationships:
            if (relation.get("parent_table_name") == table_name or
                relation.get("child_table_name") == table_name):
                rels.append(relation)
        return rels

    def get_foreign_keys(self, parent_table_name: str, child_table_name: str) -> list:
        """Get foreign keys between parent and child tables.

        Parameters
        ----------
        parent_table_name: str
            Name of the parent table.
        child_table_name: str
            Name of the child table.

        Returns
        -------
        list
            List of foreign key column names.
        """
        foreign_keys = []
        for relation in self.relationships:
            if (relation.get("parent_table_name") == parent_table_name and
                relation.get("child_table_name") == child_table_name):
                fk = relation.get("child_foreign_key")
                if fk:
                    foreign_keys.append(fk)
        return foreign_keys

    def get_root_tables(self) -> list:
        """Get all root tables (tables with no parents).

        Returns
        -------
        list
            List of root table names.
        """
        root_tables = set(self.tables.keys())
        for relation in self.relationships:
            root_tables.discard(relation.get("child_table_name"))
        return list(root_tables)

    def get_table_levels(self) -> dict:
        """Get the level of each table in the hierarchy.

        The level is determined by the length of the path from any root table.

        Returns
        -------
        dict
            Dictionary mapping table names to their levels.
        """
        root_tables = self.get_root_tables()
        table_levels = {table: 0 for table in root_tables}

        relationships = self.relationships.copy()
        max_iterations = len(self.tables) * len(self.relationships)
        iteration = 0
        
        while relationships and iteration < max_iterations:
            relationship = relationships.pop(0)
            parent = relationship.get("parent_table_name")
            child = relationship.get("child_table_name")
            
            if parent in table_levels:
                table_levels[child] = table_levels[parent] + 1
            else:
                relationships.append(relationship)
            
            iteration += 1
        
        return table_levels

    def to_dict(self) -> dict:
        """Convert metadata to dictionary format.

        Returns
        -------
        dict
            Metadata as a dictionary.
        """
        return {
            'tables': self.tables,
            'relationships': self.relationships
        }
        
    def get_column_sdtype(self, table_name: str, column_name: str) -> str:
        """Get the sdtype of a specific column.
        
        Returns
        -------
        str
            One of: 'numerical', 'categorical', 'boolean', 'datetime', 'id', etc.
        """
        table_meta = self.get_table_meta(table_name)
        if column_name not in table_meta.get('columns', {}):
            raise ValueError(f"Column '{column_name}' not found in table '{table_name}'")
        
        return table_meta['columns'][column_name].get('sdtype')

    def get_columns_by_sdtype(self, table_name: str, sdtype: str) -> list:
        """Get all columns of a specific sdtype.
        
        Parameters
        ----------
        sdtype: str
            Type to filter by: 'numerical', 'categorical', 'boolean', 'datetime', 'id'
        """
        table_meta = self.get_table_meta(table_name)
        columns = []
        
        for col_name, col_info in table_meta.get('columns', {}).items():
            if col_info.get('sdtype') == sdtype:
                columns.append(col_name)
        
        return columns

    def get_all_column_types(self, table_name: str) -> dict:
        """Get all columns with their sdtypes.
        
        Returns
        -------
        dict
            {column_name: sdtype}
        """
        table_meta = self.get_table_meta(table_name)
        return {
            col_name: col_info.get('sdtype')
            for col_name, col_info in table_meta.get('columns', {}).items()
        }
        
    def get_datetime_col_info(self, table_name: str):
        """메타데이터에서 날짜/시간 컬럼의 이름과 포맷을 찾습니다."""
        for col, info in self.tables[table_name]['columns'].items():
            if info['sdtype'] == 'datetime':
                return col, info.get('datetime_format')
        return None, None