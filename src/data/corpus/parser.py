""" Arithmetic equation parser """
import re

debug_parser = False


def parse(buffer):
    if debug_parser: print("PARSE:", buffer)
    buffer.append(None)  # end-of-stream "token"
    ret, pos = parse_expr(buffer, 0)
    if buffer[pos] is not None:
        raise BaseException('parse error: {} ^ {}'.format(buffer[:pos], buffer[pos:]))
    else:
        return ret


def accept(buffer, pos, tokens):
    if buffer[pos] in set(tokens):
        return buffer[pos], pos + 1
    else:
        raise BaseException('parse error: {} ^ {}'.format(buffer[:pos], buffer[pos:]))


# expr : expr_sum '=' expr_sum
#     ;
def parse_expr(buffer, pos):
    if debug_parser: print("expr: {} ^ {}".format(buffer[:pos], buffer[pos:]))
    a, pos = parse_expr_sum(buffer, pos)
    opp, pos = accept(buffer, pos, ['='])
    b, pos = parse_expr_sum(buffer, pos)
    return (a, opp, b), pos


# expr_sum : expr_mul (('+'|'-') expr_mul)*
#         ;
def parse_expr_sum(buffer, pos):
    if debug_parser: print("expr_sum: {} ^ {}".format(buffer[:pos], buffer[pos:]))
    ret, pos = parse_expr_mul(buffer, pos)
    while buffer[pos] == '+' or buffer[pos] == '-':
        if debug_parser: print("accept:", buffer[pos])
        opp = buffer[pos]
        pos += 1
        b, pos = parse_expr_mul(buffer, pos)
        ret = (ret, opp, b)
    return ret, pos


# expr_mul : atom (('*'|'/') atom)*
#         ;
def parse_expr_mul(buffer, pos):
    if debug_parser: print("expr_mul: {} ^ {}".format(buffer[:pos], buffer[pos:]))
    ret, pos = parse_atom(buffer, pos)
    while buffer[pos] == '*' or buffer[pos] == '/':
        if debug_parser: print("accept:", buffer[pos])
        opp = buffer[pos]
        pos += 1
        b, pos = parse_atom(buffer, pos)
        ret = (ret, opp, b)
    return ret, pos


# atom : NUMBER
#      | '(' expr_sum ')'
#      ;
def parse_atom(buffer, pos):
    if debug_parser: print("atom: {} ^ {}".format(buffer[:pos], buffer[pos:]))
    if re.match('[-+]?\d+', buffer[pos]) or buffer[pos] == 'x':
        if debug_parser: print("accept:", buffer[pos])
        return buffer[pos], pos + 1
    elif buffer[pos] == '(':
        _, pos = accept(buffer, pos, ['('])
        ret, pos = parse_expr_sum(buffer, pos)
        _, pos = accept(buffer, pos, [')'])
        return ret, pos
    else:
        raise BaseException('parse error: {} ^ {}'.format(buffer[:pos], buffer[pos:]))
