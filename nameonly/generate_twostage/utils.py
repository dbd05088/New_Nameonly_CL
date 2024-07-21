import re

def sanitize_filename(filename):
    forbidden_chars = r'[\/\\\?\%\*\:\|\"<>\.]'
    
    sanitized = re.sub(forbidden_chars, '', filename)
    
    sanitized = re.sub(r'\s+', '_', sanitized)
    
    max_length = 255
    sanitized = sanitized[:max_length]
    
    return sanitized