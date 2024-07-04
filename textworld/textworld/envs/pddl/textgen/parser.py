#!/usr/bin/env python

# CAVEAT UTILITOR
#
# This file was automatically generated by TatSu.
#
#    https://pypi.python.org/pypi/tatsu/
#
# Any changes you make to it will be overwritten the next time
# the file is generated.

from __future__ import annotations

import sys

from tatsu.buffering import Buffer
from tatsu.parsing import Parser
from tatsu.parsing import tatsumasu
from tatsu.parsing import leftrec, nomemo, isname # noqa
from tatsu.infos import ParserConfig
from tatsu.util import re, generic_main  # noqa


KEYWORDS = {}  # type: ignore


class CSGBuffer(Buffer):
    def __init__(self, text, /, config: ParserConfig = None, **settings):
        config = ParserConfig.new(
            config,
            owner=self,
            whitespace=re.compile('[\\t]+'),
            nameguard=None,
            comments_re=None,
            eol_comments_re='^(//.*|\\s*)\\n?',
            ignorecase=False,
            namechars='',
            parseinfo=False,
        )
        config = config.replace(**settings)
        super().__init__(text, config=config)


class CSGParser(Parser):
    def __init__(self, /, config: ParserConfig = None, **settings):
        config = ParserConfig.new(
            config,
            owner=self,
            whitespace=re.compile('[\\t]+'),
            nameguard=None,
            comments_re=None,
            eol_comments_re='^(//.*|\\s*)\\n?',
            ignorecase=False,
            namechars='',
            parseinfo=False,
            keywords=KEYWORDS,
            start='start',
        )
        config = config.replace(**settings)
        super().__init__(config=config)

    @tatsumasu()
    def _start_(self):  # noqa
        self._symbols_()

    @tatsumasu()
    def _tag_(self):  # noqa
        self._pattern('[\\w()/!<>\\-\\s,.]+')

    @tatsumasu()
    def _given_(self):  # noqa
        self._pattern('[^;|{}\\n\\[\\]#]+')

    @tatsumasu()
    def _statement_(self):  # noqa
        self._pattern('[^|\\[\\]{}\\n<>]+')

    @tatsumasu()
    def _Literal_(self):  # noqa
        self._pattern('[^;|"<>\\[\\]#{}]*')

    @tatsumasu('TerminalSymbol')
    def _terminalSymbol_(self):  # noqa
        with self._group():
            with self._choice():
                with self._option():
                    self._token('"')
                    self._Literal_()
                    self.name_last_node('literal')
                    self._token('"')

                    self._define(
                        ['literal'],
                        []
                    )
                with self._option():
                    self._cut()
                    self._Literal_()
                    self.name_last_node('literal')

                    self._define(
                        ['literal'],
                        []
                    )
                self._error(
                    'expecting one of: '
                    '\'"\' \'~\''
                )

    @tatsumasu('NonterminalSymbol')
    def _nonterminalSymbol_(self):  # noqa
        self._token('#')
        self._tag_()
        self.name_last_node('symbol')
        self._token('#')

        self._define(
            ['symbol'],
            []
        )

    @tatsumasu('EvalSymbol')
    def _evalSymbol_(self):  # noqa
        self._statement_()
        self.name_last_node('statement')

    @tatsumasu('ConditionalSymbol')
    def _conditionalSymbol_(self):  # noqa
        self._token('{')
        with self._group():
            with self._choice():
                with self._option():
                    self._nonterminalSymbol_()
                with self._option():
                    self._evalSymbol_()
                self._error(
                    'expecting one of: '
                    '<evalSymbol> <nonterminalSymbol>'
                )
        self.name_last_node('expression')
        with self._optional():
            self._pattern('\\s*\\|\\s*')
            self._given_()
            self.name_last_node('given')

            self._define(
                ['given'],
                []
            )
        self._token('}')

        self._define(
            ['expression', 'given'],
            []
        )

    @tatsumasu('ListSymbol')
    def _listSymbol_(self):  # noqa
        self._token('[')
        self._conditionalSymbol_()
        self.name_last_node('symbol')
        self._token(']')

        self._define(
            ['symbol'],
            []
        )

    @tatsumasu()
    def _Symbol_(self):  # noqa
        with self._choice():
            with self._option():
                self._listSymbol_()
            with self._option():
                self._conditionalSymbol_()
            with self._option():
                self._nonterminalSymbol_()
            with self._option():
                self._terminalSymbol_()
            self._error(
                'expecting one of: '
                '\'"\' \'#\' \'[\' \'{\' \'~\' <conditionalSymbol>'
                '<listSymbol> <nonterminalSymbol>'
                '<terminalSymbol>'
            )

    @tatsumasu()
    def _symbols_(self):  # noqa

        def block0():
            self._Symbol_()
        self._positive_closure(block0)


class CSGSemantics:
    def start(self, ast):  # noqa
        return ast

    def tag(self, ast):  # noqa
        return ast

    def given(self, ast):  # noqa
        return ast

    def statement(self, ast):  # noqa
        return ast

    def Literal(self, ast):  # noqa
        return ast

    def terminalSymbol(self, ast):  # noqa
        return ast

    def nonterminalSymbol(self, ast):  # noqa
        return ast

    def evalSymbol(self, ast):  # noqa
        return ast

    def conditionalSymbol(self, ast):  # noqa
        return ast

    def listSymbol(self, ast):  # noqa
        return ast

    def Symbol(self, ast):  # noqa
        return ast

    def symbols(self, ast):  # noqa
        return ast


def main(filename, **kwargs):
    if not filename or filename == '-':
        text = sys.stdin.read()
    else:
        with open(filename) as f:
            text = f.read()
    parser = CSGParser()
    return parser.parse(
        text,
        filename=filename,
        **kwargs
    )


if __name__ == '__main__':
    import json
    from tatsu.util import asjson

    ast = generic_main(main, CSGParser, name='CSG')
    data = asjson(ast)
    print(json.dumps(data, indent=2))