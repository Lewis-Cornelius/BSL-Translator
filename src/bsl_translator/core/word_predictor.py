"""
BSL Translator - Word Predictor Module
NLTK-based word completion and autocorrection.
"""
import nltk
import difflib
from typing import List, Set
import logging

logger = logging.getLogger(__name__)


class WordPredictor:
    """Word prediction and autocorrection using NLTK."""
    
    def __init__(self):
        """Initialize word predictor with NLTK dictionary."""
        self.dictionary = self._load_dictionary()
    
    def _load_dictionary(self) -> Set[str]:
        """
        Load word dictionary from NLTK.
        
        Returns:
            Set of uppercase words
        """
        try:
            nltk.download('words', quiet=True)
            from nltk.corpus import words
            word_dict = set(word.upper() for word in words.words())
            logger.info("Word dictionary loaded successfully")
            return word_dict
        except Exception as e:
            logger.warning(f"Error downloading NLTK words: {e}. Using fallback.")
            # Fallback dictionary
            return set([
                'HELLO', 'THANK', 'YOU', 'PLEASE', 'HELP', 'GOOD', 'BAD',
                'YES', 'NO', 'MAYBE', 'HOW', 'WHAT', 'WHERE', 'WHEN', 'WHO',
                'LOVE', 'SORRY', 'WELCOME', 'THANKS', 'MORNING', 'NIGHT'
            ])
    
    def get_suggestions(self, fragment: str, num: int = 3) -> List[str]:
        """
        Get word suggestions based on input fragment.
        
        Args:
            fragment: Partial word (e.g. "HEL")
            num: Number of suggestions to return
            
        Returns:
            List of suggested complete words
        """
        if not fragment:
            return []
        
        fragment = fragment.upper()
        
        # Use difflib for fuzzy matching
        matches = difflib.get_close_matches(
            fragment, 
            self.dictionary, 
            n=num, 
            cutoff=0.6
        )
        
        return matches
    
    def autocorrect(self, word: str) -> str:
        """
        Autocorrect a word if it's not in dictionary.
        
        Args:
            word: Word to correct
            
        Returns:
            Corrected word (or original if no good match)
        """
        word = word.upper()
        
        # If already in dictionary, return as-is
        if word in self.dictionary:
            return word
        
        # Try to find close match
        suggestions = self.get_suggestions(word, num=1)
        if suggestions:
            logger.debug(f"Autocorrected '{word}' to '{suggestions[0]}'")
            return suggestions[0]
        
        # No correction found
        return word
