import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories
    # Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = sorted(
        path for path in current_dir.glob("chroma_db*") if path.is_dir()
    )

    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )

            # Retrieve list of available collections from the database
            collections = client.list_collections()

            # Loop through each collection found
            for collection in collections:
                # Create unique identifier key combining directory and collection names
                backend_key = f"{chroma_dir.name}:{collection.name}"

                # Build information dictionary containing:
                # Store directory path as string
                # Store collection name
                # Create user-friendly display name
                # Get document count with fallback for unsupported operations
                try:
                    document_count = collection.count()
                except (AttributeError, NotImplementedError):
                    document_count = 0

                backend_info = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": (
                        f"{chroma_dir.name} / {collection.name} "
                        f"({document_count} documents)"
                    ),
                    "document_count": document_count,
                }
                # Add collection information to backends dictionary
                backends[backend_key] = backend_info
        except Exception as error:
            # Handle connection or access errors gracefully
            # Create fallback entry for inaccessible directories
            backend_key = f"{chroma_dir.name}:error"
            backends[backend_key] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                # Include error information in display name with truncation
                "display_name": (
                    f"{chroma_dir.name} "
                    f"(Unavailable: {str(error)[:50]})"
                ),
                # Set appropriate fallback values for missing information
                "document_count": 0,
            }

    # Return complete backends dictionary with all discovered collections
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    try:
        # Create a chomadb persistentclient
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(collection_name)
        embedding_function = collection.configuration["embedding_function"]
        collection = client.get_collection(
            collection_name,
            embedding_function=embedding_function,
        )
        # Return the collection with the collection_name
        return collection, True, None
    except Exception as error:
        return None, False, str(error)


def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable to None (represents no filtering)
    where_filter = None

    # Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter:
        mission = mission_filter.lower().replace(" ", "_")
        # If filter conditions are met, create filter dictionary with appropriate field-value pairs
        if mission not in ("all", "all_missions"):
            where_filter = {"mission": mission}

    # Execute database query with the following parameters:
    results = collection.query(
        # Pass search query in the required format
        query_texts=[query],
        # Set maximum number of results to return
        n_results=n_results,
        # Apply conditional filter (None for no filtering, dictionary for specific filtering)
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    # Return query results to caller
    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    # Initialize list with header text for context section
    context_parts = ["NASA mission context:"]
    seen_documents = set()
    source_number = 1

    # Loop through paired documents and their metadata using enumeration
    for _, (document, metadata) in enumerate(
        zip(documents, metadatas), start=1
    ):
        if document in seen_documents:
            continue
        seen_documents.add(document)

        # Extract mission information from metadata with fallback value
        mission = metadata.get("mission", "unknown")
        # Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").title()

        # Extract source information from metadata with fallback value
        source = metadata.get("source", "Unknown source")

        # Extract category information from metadata with fallback value
        category = metadata.get("document_category", "unknown")
        # Clean up category name formatting (replace underscores, capitalize)
        category = category.replace("_", " ").title()

        # Create formatted source header with index number and extracted information
        source_header = (
            f"\n[Source {source_number}] "
            f"Mission: {mission} | Category: {category} | File: {source}"
        )
        # Add source header to context parts list
        context_parts.append(source_header)

        # Check document length and truncate if necessary
        if len(document) > 2000:
            document = document[:2000] + "..."
        # Add truncated or full document content to context parts list
        context_parts.append(document)
        source_number += 1

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)
