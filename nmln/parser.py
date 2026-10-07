from pyparsing import *


ParserElement.enablePackrat()


symbol = Word(alphanums + "_")
left_parenthesis, right_parenthesis, colon, left_square, right_square, dot = map(
    Suppress, "():[]."
)
parser_atom = (
    symbol
    + left_parenthesis
    + delimitedList(symbol)
    + right_parenthesis
    + Optional(dot)
)


def atom_parser(atom_string):
    tokens = parser_atom.parseString(atom_string)
    return tokens[0], tokens[1:]
