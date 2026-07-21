"""
Integration test for the Graph-M-RAG pipeline with semantic graph construction.
This script tests that the document upload process correctly triggers both
structural and semantic graph construction.
"""
import asyncio
import requests
import tempfile
import os
from pathlib import Path

def test_upload_triggers_both_graphs():
    """
    Test that uploading a document triggers both structural and semantic graph construction
    """
    # Sample PDF file for testing (we'll create a minimal one)
    sample_pdf_content = b'%PDF-1.4\n%\xc7\xec\x80\xe4\n4 0 obj\n<<\n/Type /Catalog\n/Pages 3 0 R\n>>\nendobj\n5 0 obj\n<<\n/Producer (Sample PDF Producer)\n/Creator (Sample PDF Creator)\n/Title (Sample PDF Title)\n>>\nendobj\n3 0 obj\n<<\n/Type /Pages\n/Kids [2 0 R]\n/Count 1\n>>\nendobj\n2 0 obj\n<<\n/Type /Page\n/Parent 3 0 R\n/MediaBox [0 0 612 792]\n/Contents 1 0 R\n>>\nendobj\n1 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Test content for Graph-M-RAG integration) Tj\nET\nendstream\nendobj\n6 0 obj\n<<\n/Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n>>\nendobj\nxref\n0 7\n0000000000 65535 f \n0000000223 00000 n \n0000000437 00000 n \n0000000302 00000 n \n0000000015 00000 n \n0000000090 00000 n \n0000000553 00000 n \ntrailer\n<<\n/Size 7\n/Root 4 0 R\n/Info 5 0 R\n>>\n%%EOF'
    
    # Create a temporary PDF file
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(sample_pdf_content)
        temp_pdf_path = tmp_file.name

    try:
        # Upload the test PDF to trigger the pipeline
        print("Uploading test PDF to trigger pipeline...")
        with open(temp_pdf_path, 'rb') as pdf_file:
            response = requests.post(
                "http://localhost:9191/upload-pdf",
                files={"file": pdf_file}
            )
        
        print(f"Upload response status: {response.status_code}")
        print(f"Upload response: {response.json()}")
        
        if response.status_code != 200:
            print("Upload failed!")
            return False
        
        response_data = response.json()
        file_hash = response_data.get("file_hash")
        
        if not file_hash:
            print("No file hash returned from upload!")
            return False
            
        print(f"Uploaded document with hash: {file_hash}")
        
        # Wait a bit for processing to complete
        import time
        time.sleep(5)
        
        # Check that both structural and semantic graphs were created
        # by querying Neo4j for document-related nodes
        print("Checking for structural and semantic graph nodes in Neo4j...")
        
        # This assumes Neo4j is running with the default credentials from the project
        from neo4j import GraphDatabase
        
        neo4j_uri = "bolt://localhost:7687"
        neo4j_user = "neo4j"
        neo4j_password = "neo4j123"
        
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            # Check for structural graph elements (Document node with file_hash)
            struct_query = """
            MATCH (d:Document {file_hash: $file_hash})
            RETURN count(d) AS doc_count
            """
            struct_result = session.run(struct_query, file_hash=file_hash)
            doc_count = struct_result.single()["doc_count"]
            
            # Check for semantic graph elements (Community nodes)
            sem_query = """
            MATCH (c:Community)
            OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:Entity)
            WHERE e IS NOT NULL
            WITH c, count(e) AS entity_count
            WHERE entity_count > 0
            RETURN count(c) AS community_count
            """
            sem_result = session.run(sem_query)
            community_count = sem_result.single()["community_count"]
            
            # Check for connections between structural and semantic graphs
            conn_query = """
            MATCH (d:Document {file_hash: $file_hash})-[r:CONNECTS_TO]->(c:Community)
            RETURN count(r) AS connection_count
            """
            conn_result = session.run(conn_query, file_hash=file_hash)
            connection_count = conn_result.single()["connection_count"]
            
            print(f"Structural graph elements (Document nodes): {doc_count}")
            print(f"Semantic graph elements (Community nodes with entities): {community_count}")
            print(f"Connections between graphs: {connection_count}")
            
            driver.close()
            
            # Both graphs should have been created and connected
            success = doc_count > 0 and community_count > 0 and connection_count > 0
            
            if success:
                print("SUCCESS: Both structural and semantic graphs were created and connected!")
            else:
                print("FAILURE: One or more components are missing.")
                print(f"- Structural graph (Document nodes): {'✓' if doc_count > 0 else '✗'}")
                print(f"- Semantic graph (Community nodes): {'✓' if community_count > 0 else '✗'}")
                print(f"- Connections between graphs: {'✓' if connection_count > 0 else '✗'}")
            
            return success
            
    except Exception as e:
        print(f"Error during integration test: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


if __name__ == "__main__":
    print("Running Graph-M-RAG integration test...")
    success = test_upload_triggers_both_graphs()
    
    if success:
        print("\nIntegration test PASSED: Both graphs are being created correctly!")
    else:
        print("\nIntegration test FAILED: Issues detected in the pipeline.")