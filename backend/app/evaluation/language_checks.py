"""Conservative script/style flags, not a substitute for human translation review."""
import re
HINDI_WORDS=set('hai hain tha thi kya kaun kis kisko kab kahan kaise kyun kyon mein me ke ki ka ko se aur ya nahi nahin liye par pr yeh ye vo woh hota hoti hote sakta sakti sakte kare karta karti karna adhikar anuchhed samvidhan bharat niyam kanoon tahat'.split())
def language_issues(text,language):
    letters=[c for c in text if c.isalpha()]
    if not letters: return ['No readable language text.']
    devanagari=sum('\u0900'<=c<='\u097f' for c in letters)/len(letters)
    words=set(re.findall(r'[a-z]+',text.lower()))
    if language=='hi' and devanagari<0.35: return ['Hindi requires substantial Devanagari text.']
    if language=='hinglish':
        if devanagari>0.15: return ['Hinglish must use primarily Roman script.']
        if len(words & HINDI_WORDS)<2: return ['Possible English-only text labelled Hinglish; use natural Roman Hindi, then review.']
    if language=='en' and devanagari>0.15: return ['English row contains substantial Devanagari text.']
    return []
