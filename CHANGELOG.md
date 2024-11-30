# CHANGELOG

## [inline-citation-branch]

### Added
- **GraphQL Schema Update:**
  - Added `Citation` and `SearchResult` types to the GraphQL schema in `backend/app.py`.
    - `Citation` type includes `id` and `context` fields.
    - `SearchResult` type includes `title`, `link`, `summary`, and `citations` fields.

### Changed
- **GraphQL Query Update:**
  - Updated the `askLlm` query to return a list of `SearchResult` objects instead of a single string in `backend/app.py`.

- **Resolver Update:**
  - Modified the `resolve_ask_llm` function to handle the new return type in `backend/app.py`.

### Fixed
- **Replaced Deprecated Imports:**
  - Updated imports from `langchain` to `langchain_community` in `backend/llm_utils.py`.

- **Logger Initialization:**
  - Added logger setup to ensure proper logging in `backend/llm_utils.py`.

- **Enhanced Document Loaders:**
  - Improved metadata extraction and assignment for PDF and YouTube documents in `backend/llm_utils.py`.
  - Added exception handling for the YouTube loader in `backend/llm_utils.py`.



## Detailed Changes

### GraphQL Schema Update

**File:** `backend/app.py`

**Changes:**
- **Added `Citation` Type:**
  - Fields: `id` (String), `context` (String)
  - **Purpose:** Represents a citation within a search result, providing context for the summary.
- **Added `SearchResult` Type:**
  - Fields: `title` (String), `link` (String), `summary` (String), `citations` (List of `Citation`)
  - **Purpose:** Represents a search result with a title, link, summary, and associated citations.

**Why:**
- To support the new return type for the `askLlm` query, which now returns a list of `SearchResult` objects instead of a single string.
- **Impact:** Allows the frontend to receive structured search results with titles, links, summaries, and citations, enhancing the user experience.

### GraphQL Query Update

**File:** `backend/app.py`

**Changes:**
- **Updated `askLlm` Query:**
  - Changed the return type from a single string to a list of `SearchResult` objects.

**Why:**
- To handle the new return type for the `askLlm` query.
- **Impact:** Ensures that the backend correctly processes and returns the updated response format, which includes a list of `SearchResult` objects.

### Resolver Update

**File:** `backend/app.py`

**Changes:**
- **Modified `resolve_ask_llm` Function:**
  - Updated to handle the new return type for the `askLlm` query.

**Why:**
- To handle the new return type for the `askLlm` query.
- **Impact:** Ensures that the backend correctly processes and returns the updated response format, which includes a list of `SearchResult` objects.

### Enhanced Document Loaders

**File:** `backend/llm_utils.py`

**Changes:**
- **Replaced Deprecated Imports:**
  - Updated imports from `langchain` to `langchain_community`.

**Why:**
- To comply with the latest `langchain` library updates and avoid future compatibility issues.
- **Impact:** Ensures that the codebase remains up-to-date and functional with the latest library versions.

**Changes:**
- **Logger Initialization:**
  - Added logger setup to ensure proper logging.

**Why:**
- To enable proper logging throughout the application for better debugging and monitoring.
- **Impact:** Provides detailed logs for document loading, processing, and error handling.

**Changes:**
- **Enhanced Document Loaders:**
  - Improved metadata extraction and assignment for PDF and YouTube documents.
  - Added exception handling for the YouTube loader.

**Why:**
- To improve metadata extraction and assignment for PDF and YouTube documents, and to add exception handling for the YouTube loader.
- **Impact:** Ensures that documents are loaded with accurate metadata and that errors are gracefully handled, preventing application crashes.


.

---

## Setup Instructions for New Branch

### Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>