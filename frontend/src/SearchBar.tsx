import React, { useState, useRef, useEffect } from 'react';
import { Search, Send } from 'react-feather';

interface Citation {
  id: string;
  context: string;
}

interface SearchResult {
  title: string;
  link: string;
  summary: string;
  citations: Citation[];
}


const SearchBar = () => {
  const [inputValue, setInputValue] = useState('');
  const [conversation, setConversation] = useState<{ type: string; content: string; source?: string }[]>([]);
  const conversationEndRef = useRef<HTMLDivElement>(null); //hook to create a reference to the end of the conversation container.
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  //Hook to scroll the container to the bottom whenever the conversation array changes.
  useEffect(() => {
    if (conversationEndRef.current) {
      conversationEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversation]);


  //Function to handle the search query
  const handleSearch = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/graphql', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: `
            query Search($prompt: String!) {
              askLlm(prompt: $prompt) {
                title
                link
                summary
                citations {
                  id
                  context
                }
              }
            }
          `,
          variables: {
            prompt: inputValue
          }
        }),
      });
  
      const data = await response.json();
      setResults(data.data.askLlm);
      setInputValue('');
    } catch (error) {
      console.error(error);
    } finally {
      setIsLoading(false); // Set loading to false
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
    {/* Search Form */}
    <form onSubmit={handleSearch} className="flex gap-2 mb-4">
      <input
        type="search"
        placeholder="Search"
        value={inputValue}
        onChange={handleInputChange}
        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      <button
        type="submit"
        className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
      >
        Search
      </button>
    </form>

    {/* Results Listing */}
    {/* Loading Indicator */}
    {isLoading && (
      <div className="flex justify-center items-center my-4">
        <svg
          className="animate-spin h-8 w-8 text-blue-500"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          ></circle>
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8v8H4z"
          ></path>
        </svg>
        <span className="ml-2 text-blue-500">Searching...</span>
      </div>
    ) }

    {/* Results */}
    {!isLoading && results.length > 0 && (<div className="space-y-6" >
      {results.map((result, index) => (
        <div key={index} className="border rounded-lg p-6 hover:shadow-lg transition-shadow">
          <h2 className="text-xl font-semibold mb-2">
            <a
              href={result.link}
              className="text-blue-600 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              {result.title}
            </a>
          </h2>

          <p className="text-gray-700">
            {result.summary}
            {result.citations.map(citation => (
              <span
                key={citation.id}
                className="inline-block mx-1 cursor-help relative group"
              >
                <span className="text-blue-500">[{citation.id}]</span>
                <span className="invisible group-hover:visible absolute bottom-full left-1/2 transform -translate-x-1/2 bg-black text-white text-sm rounded p-2 w-64">
                  {citation.context}
                </span>
              </span>
            ))}
          </p>
        </div>
      ))}
    </div>)}
    
  </div>
  );
};

export default SearchBar;