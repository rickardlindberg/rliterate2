from setuptools import setup

setup(
    name='rlgui_lexer',
    version='0.1',
    packages=['rlgui_lexer'],
    entry_points={
        'pygments.lexers': ['rlgui_lexer=rlgui_lexer:RLGuiLexer'],
    },
    zip_safe=False,
)
