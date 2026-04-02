"""
Document Index Package - Neo4j Graph Management for PDF Documents

This package provides functionality to create and manage document graphs in Neo4j
based on MinerU PDF processing results.
"""

from .manager import Manager, ManagerConfig, Neo4jConnection
from .dtype import Document, Region, Style, BBox
from .neo4j_service import DocumentIndexService, create_neo4j_graph

__all__ = [
    'Manager',
    'ManagerConfig',
    'Neo4jConnection',
    'Document',
    'Region',
    'Style',
    'BBox',
    'DocumentIndexService',
    'create_neo4j_graph'
]