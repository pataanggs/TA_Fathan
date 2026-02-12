"""
Spell corrector for post-processing.
Uses Levenshtein distance to correct out-of-vocabulary words
to the nearest valid word in the Minangkabau dictionary.

Uses frequency-weighted tie-breaking: when multiple candidates
have the same edit distance, the most frequent word is preferred.
"""

from typing import Dict, Optional, Set, Tuple, Union

from config import DICTIONARY_CONFIG


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.
    Uses optimized single-row dynamic programming.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (number of insertions, deletions, substitutions)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost: 0 if characters match, 1 if substitution needed
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def find_best_correction(
    word: str,
    dictionary: Union[Set[str], Dict[str, int]],
    max_distance: int = None,
) -> Tuple[Optional[str], int]:
    """
    Find the closest dictionary word within max_distance edits.
    If dictionary is a Dict[str, int] (word→frequency), uses frequency
    for tie-breaking among equal-distance candidates.

    Args:
        word: Input word to correct
        dictionary: Set of valid words, or Dict of word→frequency
        max_distance: Maximum allowed edit distance

    Returns:
        Tuple of (best_match or None, distance)
    """
    if max_distance is None:
        max_distance = DICTIONARY_CONFIG["max_edit_distance"]

    # Word already in dictionary — no correction needed
    if word in dictionary:
        return word, 0

    best_match = None
    best_dist = max_distance + 1
    best_freq = 0

    # Determine if we have frequency info
    has_freq = isinstance(dictionary, dict)

    for dict_word in dictionary:
        # Quick filter: skip if length difference exceeds max_distance
        if abs(len(dict_word) - len(word)) > max_distance:
            continue

        dist = levenshtein_distance(word, dict_word)

        if dist < best_dist:
            best_dist = dist
            best_match = dict_word
            best_freq = dictionary[dict_word] if has_freq else 0
        elif dist == best_dist and has_freq:
            # Tie-breaking: prefer more frequent word
            freq = dictionary[dict_word]
            if freq > best_freq:
                best_match = dict_word
                best_freq = freq

        # Early exit if perfect neighbor found (distance 1)
        if dist == 1 and not has_freq:
            return best_match, 1

    if best_dist <= max_distance:
        return best_match, best_dist

    return None, best_dist


def correct_sentence(
    sentence: str,
    dictionary: Union[Set[str], Dict[str, int]],
    max_distance: int = None,
) -> Tuple[str, list]:
    """
    Correct all words in a sentence using the dictionary.

    Args:
        sentence: Normalized input sentence
        dictionary: Set of valid words, or Dict of word→frequency
        max_distance: Maximum allowed edit distance

    Returns:
        Tuple of (corrected_sentence, list_of_corrections)
    """
    if max_distance is None:
        max_distance = DICTIONARY_CONFIG["max_edit_distance"]

    min_len = DICTIONARY_CONFIG["min_word_length"]
    words = sentence.split()
    corrected_words = []
    corrections = []

    for word in words:
        # Skip short words (articles, particles — less likely to be misspelled)
        if len(word) < min_len:
            corrected_words.append(word)
            continue

        best_match, dist = find_best_correction(word, dictionary, max_distance)

        if best_match is not None and dist > 0:
            corrections.append({
                "original": word,
                "corrected": best_match,
                "distance": dist,
            })
            corrected_words.append(best_match)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words), corrections


if __name__ == "__main__":
    print("=" * 60)
    print("SPELL CORRECTOR — Test Cases")
    print("=" * 60)

    # Simulated dictionary (subset)
    test_dict = {
        "manusia", "sadonyo", "lahia", "dunia", "mambao",
        "hak", "dan", "kamardekaan", "mandasar", "nan",
        "samo", "indak", "dapek", "dipisahkan", "bantulah",
        "mawujudkan", "mampatahankannyo", "untuak", "diri",
        "awak", "surang", "atau", "pun", "sasamo", "umat",
        "tabukti", "mangakibatkan", "parilaku", "biadab",
        "malukoi", "nurani", "sabagai", "cito", "paliang",
        "tinggi", "dari", "urang",
    }

    test_sentences = [
        "bantula mau ujutkan dan mampatahankannyo",
        "parilaku biadap nan sangat malukoi",
        "sabagai cito cito paliang tinggi dari sadonyo urang",
    ]

    for sent in test_sentences:
        corrected, corrections = correct_sentence(sent, test_dict)
        print(f"\nOriginal:  {sent}")
        print(f"Corrected: {corrected}")
        if corrections:
            for c in corrections:
                print(f"  → '{c['original']}' → '{c['corrected']}' (dist={c['distance']})")
        else:
            print("  → No corrections needed")
