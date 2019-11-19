from pygments.lexer import RegexLexer
from pygments.token import *

class RLGuiLexer(RegexLexer):

    name = 'RLGui'
    aliases = ['rlgui']
    filenames = ['*.rlgui']

    tokens = {
        'root': [
            (r'"', String, "string"),
            (r'[=]', Name.Builtin),
            (r'\b(vscroll|hscroll|scroll|frame|panel|loop|if|else|True|False|None|cache_limit)\b', Keyword),
            (r'#(\w+([.]\w+)*)?', Name.Class),
            (r'[$]\w*', Name.Builtin),
            (r'@\w+', Name.Exception),
            (r'%\w+', Comment),
            (r'.', Text),
        ],
        'string': [
            (r'[^"\\]+', String),
            (r'"', String, "#pop"),
        ],
    }

