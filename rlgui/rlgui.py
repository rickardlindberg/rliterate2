#!/usr/bin/env python2

from collections import defaultdict
import sys

try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO

def rlmeta_vm(instructions, labels, start_rule, stream):
    label_counter = 0
    last_action = _ConstantSemanticAction(None)
    pc = labels[start_rule]
    call_backtrack_stack = []
    stream, pos, stream_pos_stack = (stream, 0, [])
    scope, scope_stack = (None, [])
    fail_message = None
    latest_fail_message, latest_fail_pos = (None, tuple())
    memo = {}
    while True:
        name, arg1, arg2 = instructions[pc]
        if name == "PUSH_SCOPE":
            scope_stack.append(scope)
            scope = {}
            pc += 1
            continue
        elif name == "BACKTRACK":
            call_backtrack_stack.append((labels[arg1], pos, len(stream_pos_stack), len(scope_stack)))
            pc += 1
            continue
        elif name == "CALL":
            key = (arg1, tuple([x[1] for x in stream_pos_stack]+[pos]))
            if key in memo:
                if memo[key][0] is None:
                    fail_message = memo[key][1]
                else:
                    last_action, stream_pos_stack = memo[key]
                    stream_pos_stack = stream_pos_stack[:]
                    stream, pos = stream_pos_stack.pop()
                    pc += 1
                    continue
            else:
                call_backtrack_stack.append((pc+1, key))
                pc = labels[arg1]
                continue
        elif name == "MATCH_CHARSEQ":
            for char in arg1:
                if pos >= len(stream) or stream[pos] != char:
                    fail_message = ("expected {!r}", char)
                    break
                pos += 1
            else:
                last_action = _ConstantSemanticAction(arg1)
                pc += 1
                continue
        elif name == "COMMIT":
            call_backtrack_stack.pop()
            pc = labels[arg1]
            continue
        elif name == "POP_SCOPE":
            scope = scope_stack.pop()
            pc += 1
            continue
        elif name == "RETURN":
            if len(call_backtrack_stack) == 0:
                return last_action.eval()
            pc, key = call_backtrack_stack.pop()
            memo[key] = (last_action, stream_pos_stack+[(stream, pos)])
            continue
        elif name == "LIST_APPEND":
            scope.append(last_action)
            pc += 1
            continue
        elif name == "BIND":
            scope[arg1] = last_action
            pc += 1
            continue
        elif name == "ACTION":
            last_action = _UserSemanticAction(arg1, scope)
            pc += 1
            continue
        elif name == "MATCH_RANGE":
            if pos >= len(stream) or not (arg1 <= stream[pos] <= arg2):
                fail_message = ("expected range {!r}-{!r}", arg1, arg2)
            else:
                last_action = _ConstantSemanticAction(stream[pos])
                pos += 1
                pc += 1
                continue
        elif name == "LIST_START":
            scope_stack.append(scope)
            scope = []
            pc += 1
            continue
        elif name == "LIST_END":
            last_action = _UserSemanticAction(lambda xs: [x.eval() for x in xs], scope)
            scope = scope_stack.pop()
            pc += 1
            continue
        elif name == "MATCH_ANY":
            if pos >= len(stream):
                fail_message = ("expected any",)
            else:
                last_action = _ConstantSemanticAction(stream[pos])
                pos += 1
                pc += 1
                continue
        elif name == "PUSH_STREAM":
            if pos >= len(stream) or not isinstance(stream[pos], list):
                fail_message = ("expected list",)
            else:
                stream_pos_stack.append((stream, pos))
                stream = stream[pos]
                pos = 0
                pc += 1
                continue
        elif name == "POP_STREAM":
            if pos < len(stream):
                fail_message = ("expected end of list",)
            else:
                stream, pos = stream_pos_stack.pop()
                pos += 1
                pc += 1
                continue
        elif name == "MATCH_CALL_RULE":
            if pos >= len(stream):
                fail_message = ("expected any",)
            else:
                fn_name = str(stream[pos])
                key = (fn_name, tuple([x[1] for x in stream_pos_stack]+[pos]))
                if key in memo:
                    if memo[key][0] is None:
                        fail_message = memo[key][1]
                    else:
                        last_action, stream_pos_stack = memo[key]
                        stream_pos_stack = stream_pos_stack[:]
                        stream, pos = stream_pos_stack.pop()
                        pc += 1
                        continue
                else:
                    call_backtrack_stack.append((pc+1, key))
                    pc = labels[fn_name]
                    pos += 1
                    continue
        elif name == "FAIL":
            fail_message = (arg1,)
        elif name == "LABEL":
            last_action = _ConstantSemanticAction(label_counter)
            label_counter += 1
            pc += 1
            continue
        elif name == "MATCH_STRING":
            if pos >= len(stream) or stream[pos] != arg1:
                fail_message = ("expected {!r}", arg1)
            else:
                last_action = _ConstantSemanticAction(arg1)
                pos += 1
                pc += 1
                continue
        else:
            raise Exception("unknown instruction {}".format(name))
        fail_pos = tuple([x[1] for x in stream_pos_stack]+[pos])
        if fail_pos >= latest_fail_pos:
            latest_fail_message = fail_message
            latest_fail_pos = fail_pos
        call_backtrack_entry = tuple()
        while call_backtrack_stack:
            call_backtrack_entry = call_backtrack_stack.pop()
            if len(call_backtrack_entry) == 4:
                break
            else:
                _, key = call_backtrack_entry
                memo[key] = (None, fail_message)
        if len(call_backtrack_entry) != 4:
            fail_pos = list(latest_fail_pos)
            fail_stream = stream_pos_stack[0][0] if stream_pos_stack else stream
            while len(fail_pos) > 1:
                fail_stream = fail_stream[fail_pos.pop(0)]
            raise _MatchError(latest_fail_message, fail_pos[0], fail_stream)
        (pc, pos, stream_stack_len, scope_stack_len) = call_backtrack_entry
        if len(stream_pos_stack) > stream_stack_len:
            stream = stream_pos_stack[stream_stack_len][0]
        stream_pos_stack = stream_pos_stack[:stream_stack_len]
        if len(scope_stack) > scope_stack_len:
            scope = scope_stack[scope_stack_len]
        scope_stack = scope_stack[:scope_stack_len]

class _Grammar(object):

    def run(self, rule_name, input_object):
        if isinstance(input_object, basestring):
            stream = input_object
        else:
            stream = [input_object]
        result = rlmeta_vm(self._instructions, self._labels, rule_name, stream)
        if isinstance(result, _Builder):
            return result.build_string()
        else:
            return result

class _Builder(object):

    def build_string(self):
        output = _Output()
        self.write(output)
        return output.value

    @classmethod
    def create(self, item):
        if isinstance(item, _Builder):
            return item
        elif isinstance(item, list):
            return _ListBuilder([_Builder.create(x) for x in item])
        else:
            return _AtomBuilder(item)

class _Output(object):

    def __init__(self):
        self.buffer = StringIO()
        self.indentation = 0
        self.on_newline = True

    @property
    def value(self):
        return self.buffer.getvalue()

    def write(self, value):
        for ch in value:
            is_linebreak = ch == "\n"
            if self.indentation and self.on_newline and not is_linebreak:
                self.buffer.write("    "*self.indentation)
            self.buffer.write(ch)
            self.on_newline = is_linebreak

class _ListBuilder(_Builder):

    def __init__(self, builders):
        self.builders = builders

    def write(self, output):
        for builder in self.builders:
            builder.write(output)

class _AtomBuilder(_Builder):

    def __init__(self, atom):
        self.atom = atom

    def write(self, output):
        output.write(str(self.atom))

class _IndentBuilder(_Builder):

    def write(self, output):
        output.indentation += 1

class _DedentBuilder(_Builder):

    def write(self, output):
        output.indentation -= 1

class _ConstantSemanticAction(object):

    def __init__(self, value):
        self.value = value

    def eval(self):
        return self.value

class _UserSemanticAction(object):

    def __init__(self, fn, scope):
        self.fn = fn
        self.scope = scope

    def eval(self):
        return self.fn(self.scope)

class _MatchError(Exception):

    def __init__(self, message, pos, stream):
        Exception.__init__(self)
        self.message = message
        self.pos = pos
        self.stream = stream

    def describe(self):
        message = ""
        if isinstance(self.stream, basestring):
            before = self.stream[:self.pos].splitlines()
            after = self.stream[self.pos:].splitlines()
            for context_before in before[-4:-1]:
                message += self._context(context_before)
            message += self._context(before[-1], after[0])
            message += self._arrow(len(before[-1]))
            for context_after in after[1:4]:
                message += self._context(context_after)
        else:
            message += self._context("[")
            for context_before in self.stream[:self.pos]:
                message += self._context("  ", repr(context_before), ",")
            message += self._context("  ", repr(self.stream[self.pos]), ",")
            message += self._arrow(2)
            for context_after in self.stream[self.pos+1:]:
                message += self._context("  ", repr(context_after), ",")
            message += self._context("]")
        message += "Error: "
        message += self.message[0].format(*self.message[1:])
        message += "\n"
        return message

    def _context(self, *args):
        return "> {}\n".format("".join(args))

    def _arrow(self, lenght):
        return "--{}^\n".format("-"*lenght)

def _makeList(items):
    def _addItem(depth, item):
        if depth == 0:
            result.append(item)
        else:
            for subitem in item:
                _addItem(depth-1, subitem)
    result = []
    for depth, item in items:
        _addItem(depth, item)
    return result

class GuiParser(_Grammar):

    def __init__(self):
        self._instructions = i = []
        self._labels = l = {}
        def I(name, x=None, y=None):
            i.append((name, x, y))
        def LABEL(name):
            l[name] = len(i)
        LABEL('widget')
        I('PUSH_SCOPE')
        I('CALL', 'container')
        I('BIND', 'container')
        I('CALL', 'WS')
        I('CALL', 'NAME')
        I('BIND', 'name')
        I('CALL', 'layout')
        I('BIND', 'layout')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '{')
        I('CALL', 'widgetBody')
        I('BIND', 'body')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '}')
        I('LIST_START')
        LABEL(0)
        I('BACKTRACK', 1)
        I('MATCH_CHARSEQ', ' ')
        I('LIST_APPEND')
        I('COMMIT', 0)
        LABEL(1)
        I('LIST_END')
        I('BACKTRACK', 2)
        I('MATCH_CHARSEQ', '\n')
        I('COMMIT', 3)
        LABEL(2)
        LABEL(3)
        I('LIST_START')
        LABEL(4)
        I('BACKTRACK', 5)
        I('MATCH_ANY')
        I('LIST_APPEND')
        I('COMMIT', 4)
        LABEL(5)
        I('LIST_END')
        I('BIND', 'verbatim')
        I('ACTION', lambda scope: _makeList([(0, 'widget'), (0, scope['name'].eval()), (0, scope['container'].eval()), (0, scope['layout'].eval()), (0, extract(scope['body'].eval(), 'prop')), (0, extract(scope['body'].eval(), 'instance')), (0, join(scope['verbatim'].eval()))]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('container')
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('BACKTRACK', 8)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', 'frame')
        I('POP_SCOPE')
        I('COMMIT', 9)
        LABEL(8)
        I('BACKTRACK', 6)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', 'panel')
        I('POP_SCOPE')
        I('COMMIT', 7)
        LABEL(6)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', 'vscroll')
        I('POP_SCOPE')
        LABEL(7)
        LABEL(9)
        I('BIND', 'x')
        I('CALL', 'WB')
        I('ACTION', lambda scope: _makeList([(0, 'container'), (0, scope['x'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('layout')
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('BACKTRACK', 10)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', '%layout_rows')
        I('POP_SCOPE')
        I('COMMIT', 11)
        LABEL(10)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', '%layout_columns')
        I('POP_SCOPE')
        LABEL(11)
        I('BIND', 'x')
        I('CALL', 'WB')
        I('ACTION', lambda scope: _makeList([(0, 'layout'), (0, scope['x'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('widgetBody')
        I('PUSH_SCOPE')
        I('LIST_START')
        LABEL(14)
        I('BACKTRACK', 15)
        I('BACKTRACK', 12)
        I('PUSH_SCOPE')
        I('CALL', 'instance')
        I('POP_SCOPE')
        I('COMMIT', 13)
        LABEL(12)
        I('PUSH_SCOPE')
        I('CALL', 'prop')
        I('POP_SCOPE')
        LABEL(13)
        I('LIST_APPEND')
        I('COMMIT', 14)
        LABEL(15)
        I('LIST_END')
        I('BIND', 'xs')
        I('ACTION', lambda scope: partition(scope['xs'].eval()))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('instance')
        I('BACKTRACK', 18)
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', 'loop')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '(')
        I('CALL', 'expr')
        I('BIND', 'x')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', ')')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '{')
        I('LIST_START')
        LABEL(16)
        I('BACKTRACK', 17)
        I('CALL', 'instanceLoop')
        I('LIST_APPEND')
        I('COMMIT', 16)
        LABEL(17)
        I('LIST_END')
        I('BIND', 'xs')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '}')
        I('ACTION', lambda scope: _makeList([(0, 'instance'), (0, '%loop'), (0, scope['x'].eval()), (1, scope['xs'].eval())]))
        I('POP_SCOPE')
        I('COMMIT', 19)
        LABEL(18)
        I('PUSH_SCOPE')
        I('CALL', 'instanceLoop')
        I('POP_SCOPE')
        LABEL(19)
        I('RETURN')
        LABEL('instanceLoop')
        I('BACKTRACK', 22)
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '%space')
        I('CALL', 'WB')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '[')
        I('CALL', 'expr')
        I('BIND', 'x')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', ']')
        I('ACTION', lambda scope: _makeList([(0, 'instance'), (0, '%space'), (0, scope['x'].eval())]))
        I('POP_SCOPE')
        I('COMMIT', 23)
        LABEL(22)
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('CALL', 'NAME')
        I('BIND', 'x')
        I('MATCH_CHARSEQ', '(')
        I('LIST_START')
        LABEL(20)
        I('BACKTRACK', 21)
        I('CALL', 'instanceBody')
        I('LIST_APPEND')
        I('COMMIT', 20)
        LABEL(21)
        I('LIST_END')
        I('BIND', 'xs')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', ')')
        I('ACTION', lambda scope: _makeList([(0, 'instance'), (0, scope['x'].eval()), (0, scope['xs'].eval())]))
        I('POP_SCOPE')
        LABEL(23)
        I('RETURN')
        LABEL('instanceBody')
        I('BACKTRACK', 30)
        I('PUSH_SCOPE')
        I('CALL', 'layoutAttribute')
        I('POP_SCOPE')
        I('COMMIT', 31)
        LABEL(30)
        I('BACKTRACK', 28)
        I('PUSH_SCOPE')
        I('CALL', 'loopvarExplode')
        I('POP_SCOPE')
        I('COMMIT', 29)
        LABEL(28)
        I('BACKTRACK', 26)
        I('PUSH_SCOPE')
        I('CALL', 'propExplode')
        I('POP_SCOPE')
        I('COMMIT', 27)
        LABEL(26)
        I('BACKTRACK', 24)
        I('PUSH_SCOPE')
        I('CALL', 'propAssign')
        I('POP_SCOPE')
        I('COMMIT', 25)
        LABEL(24)
        I('PUSH_SCOPE')
        I('CALL', 'handler')
        I('POP_SCOPE')
        LABEL(25)
        LABEL(27)
        LABEL(29)
        LABEL(31)
        I('RETURN')
        LABEL('layoutAttribute')
        I('BACKTRACK', 34)
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '%proportion[')
        I('CALL', 'number')
        I('BIND', 'x')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', ']')
        I('ACTION', lambda scope: _makeList([(0, 'layoutAttribute'), (0, 'proportion'), (0, scope['x'].eval())]))
        I('POP_SCOPE')
        I('COMMIT', 35)
        LABEL(34)
        I('BACKTRACK', 32)
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '%align[')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', 'EXPAND')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', ']')
        I('ACTION', lambda scope: _makeList([(0, 'layoutAttribute'), (0, 'align'), (0, 'expand')]))
        I('POP_SCOPE')
        I('COMMIT', 33)
        LABEL(32)
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '%margin[')
        I('CALL', 'expr')
        I('BIND', 'x')
        I('MATCH_CHARSEQ', ',')
        I('CALL', 'margins')
        I('BIND', 'y')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', ']')
        I('ACTION', lambda scope: _makeList([(0, 'layoutAttribute'), (0, 'margin'), (0, scope['x'].eval()), (1, scope['y'].eval())]))
        I('POP_SCOPE')
        LABEL(33)
        LABEL(35)
        I('RETURN')
        LABEL('margins')
        I('PUSH_SCOPE')
        I('CALL', 'margin')
        I('BIND', 'x')
        I('LIST_START')
        LABEL(36)
        I('BACKTRACK', 37)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', '|')
        I('CALL', 'margin')
        I('POP_SCOPE')
        I('LIST_APPEND')
        I('COMMIT', 36)
        LABEL(37)
        I('LIST_END')
        I('BIND', 'xs')
        I('ACTION', lambda scope: _makeList([(0, scope['x'].eval()), (1, scope['xs'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('margin')
        I('PUSH_SCOPE')
        I('BACKTRACK', 38)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', 'TOP')
        I('POP_SCOPE')
        I('COMMIT', 39)
        LABEL(38)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', 'BOTTOM')
        I('POP_SCOPE')
        LABEL(39)
        I('BIND', 'x')
        I('ACTION', lambda scope: _makeList([(0, 'marginSide'), (0, scope['x'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('propExplode')
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('CALL', 'PropChain')
        I('BIND', 'x')
        I('ACTION', lambda scope: _makeList([(0, 'propExplode'), (0, scope['x'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('propAssign')
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('CALL', 'NAME')
        I('BIND', 'x')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '=')
        I('CALL', 'expr')
        I('BIND', 'y')
        I('ACTION', lambda scope: _makeList([(0, 'propAssign'), (0, scope['x'].eval()), (0, scope['y'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('handler')
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '@')
        I('CALL', 'NAME')
        I('BIND', 'x')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '=')
        I('CALL', 'expr')
        I('BIND', 'y')
        I('ACTION', lambda scope: _makeList([(0, 'handler'), (0, scope['x'].eval()), (0, scope['y'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('expr')
        I('PUSH_SCOPE')
        I('CALL', 'expr1')
        I('BIND', 'x')
        I('LIST_START')
        LABEL(40)
        I('BACKTRACK', 41)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', '.')
        I('CALL', 'expr1')
        I('POP_SCOPE')
        I('LIST_APPEND')
        I('COMMIT', 40)
        LABEL(41)
        I('LIST_END')
        I('BIND', 'xs')
        I('ACTION', lambda scope: _makeList([(0, 'chain'), (0, scope['x'].eval()), (1, scope['xs'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('expr1')
        I('PUSH_SCOPE')
        I('BACKTRACK', 50)
        I('PUSH_SCOPE')
        I('CALL', 'string')
        I('POP_SCOPE')
        I('COMMIT', 51)
        LABEL(50)
        I('BACKTRACK', 48)
        I('PUSH_SCOPE')
        I('CALL', 'call')
        I('POP_SCOPE')
        I('COMMIT', 49)
        LABEL(48)
        I('BACKTRACK', 46)
        I('PUSH_SCOPE')
        I('CALL', 'number')
        I('POP_SCOPE')
        I('COMMIT', 47)
        LABEL(46)
        I('BACKTRACK', 44)
        I('PUSH_SCOPE')
        I('CALL', 'propRef')
        I('POP_SCOPE')
        I('COMMIT', 45)
        LABEL(44)
        I('BACKTRACK', 42)
        I('PUSH_SCOPE')
        I('CALL', 'identifier')
        I('POP_SCOPE')
        I('COMMIT', 43)
        LABEL(42)
        I('PUSH_SCOPE')
        I('CALL', 'loopvarRef')
        I('POP_SCOPE')
        LABEL(43)
        LABEL(45)
        LABEL(47)
        LABEL(49)
        LABEL(51)
        I('POP_SCOPE')
        I('RETURN')
        LABEL('string')
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('CALL', 'STRING')
        I('BIND', 'x')
        I('ACTION', lambda scope: _makeList([(0, 'string'), (0, scope['x'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('call')
        I('PUSH_SCOPE')
        I('BACKTRACK', 52)
        I('PUSH_SCOPE')
        I('CALL', 'identifier')
        I('POP_SCOPE')
        I('COMMIT', 53)
        LABEL(52)
        I('PUSH_SCOPE')
        I('CALL', 'propRef')
        I('POP_SCOPE')
        LABEL(53)
        I('BIND', 'x')
        I('MATCH_CHARSEQ', '(')
        I('LIST_START')
        LABEL(54)
        I('BACKTRACK', 55)
        I('CALL', 'expr')
        I('LIST_APPEND')
        I('COMMIT', 54)
        LABEL(55)
        I('LIST_END')
        I('BIND', 'xs')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', ')')
        I('ACTION', lambda scope: _makeList([(0, 'call'), (0, scope['x'].eval()), (1, scope['xs'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('number')
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('CALL', 'NUMBER')
        I('BIND', 'x')
        I('ACTION', lambda scope: _makeList([(0, 'number'), (0, scope['x'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('propRef')
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('CALL', 'PropChain')
        I('BIND', 'x')
        I('ACTION', lambda scope: _makeList([(0, 'propRef'), (0, scope['x'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('identifier')
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('CALL', 'NAME')
        I('BIND', 'x')
        I('ACTION', lambda scope: _makeList([(0, 'identifier'), (0, scope['x'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('loopvarRef')
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', '$')
        I('CALL', 'NAME')
        I('BIND', 'x')
        I('ACTION', lambda scope: _makeList([(0, 'loopvarRef'), (0, scope['x'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('loopvarExplode')
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', '$')
        I('BACKTRACK', 56)
        I('PUSH_SCOPE')
        I('CALL', 'NAME')
        I('POP_SCOPE')
        I('COMMIT', 57)
        LABEL(56)
        I('PUSH_SCOPE')
        I('ACTION', lambda scope: '')
        I('POP_SCOPE')
        LABEL(57)
        I('BIND', 'x')
        I('ACTION', lambda scope: _makeList([(0, 'loopvarExplode'), (0, scope['x'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('prop')
        I('PUSH_SCOPE')
        I('CALL', 'WS')
        I('CALL', 'NAME')
        I('BIND', 'x')
        I('CALL', 'WS')
        I('MATCH_CHARSEQ', '=')
        I('CALL', 'expr')
        I('BIND', 'y')
        I('ACTION', lambda scope: _makeList([(0, 'prop'), (0, scope['x'].eval()), (0, scope['y'].eval())]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('PropChain')
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', '#')
        I('CALL', 'NAME')
        I('BIND', 'x')
        I('LIST_START')
        LABEL(58)
        I('BACKTRACK', 59)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', '.')
        I('CALL', 'NAME')
        I('POP_SCOPE')
        I('LIST_APPEND')
        I('COMMIT', 58)
        LABEL(59)
        I('LIST_END')
        I('BIND', 'xs')
        I('CALL', 'WB')
        I('ACTION', lambda scope: join(_makeList([(0, scope['x'].eval()), (1, scope['xs'].eval())]), '.'))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('NUMBER')
        I('PUSH_SCOPE')
        I('BACKTRACK', 60)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', '-')
        I('POP_SCOPE')
        I('COMMIT', 61)
        LABEL(60)
        I('PUSH_SCOPE')
        I('ACTION', lambda scope: '')
        I('POP_SCOPE')
        LABEL(61)
        I('BIND', 'x')
        I('CALL', 'DIGIT')
        I('BIND', 'y')
        I('LIST_START')
        LABEL(62)
        I('BACKTRACK', 63)
        I('CALL', 'DIGIT')
        I('LIST_APPEND')
        I('COMMIT', 62)
        LABEL(63)
        I('LIST_END')
        I('BIND', 'ys')
        I('ACTION', lambda scope: int(join(_makeList([(0, scope['x'].eval()), (0, scope['y'].eval()), (1, scope['ys'].eval())]))))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('DIGIT')
        I('PUSH_SCOPE')
        I('MATCH_RANGE', '0', '9')
        I('POP_SCOPE')
        I('RETURN')
        LABEL('NAME')
        I('PUSH_SCOPE')
        I('CALL', 'NAMESTART')
        I('BIND', 'x')
        I('LIST_START')
        LABEL(64)
        I('BACKTRACK', 65)
        I('CALL', 'NAMECHAR')
        I('LIST_APPEND')
        I('COMMIT', 64)
        LABEL(65)
        I('LIST_END')
        I('BIND', 'xs')
        I('ACTION', lambda scope: join(_makeList([(0, scope['x'].eval()), (1, scope['xs'].eval())])))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('NAMESTART')
        I('BACKTRACK', 68)
        I('PUSH_SCOPE')
        I('MATCH_RANGE', 'a', 'z')
        I('POP_SCOPE')
        I('COMMIT', 69)
        LABEL(68)
        I('BACKTRACK', 66)
        I('PUSH_SCOPE')
        I('MATCH_RANGE', 'A', 'Z')
        I('POP_SCOPE')
        I('COMMIT', 67)
        LABEL(66)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', '_')
        I('POP_SCOPE')
        LABEL(67)
        LABEL(69)
        I('RETURN')
        LABEL('NAMECHAR')
        I('BACKTRACK', 70)
        I('PUSH_SCOPE')
        I('CALL', 'NAMESTART')
        I('POP_SCOPE')
        I('COMMIT', 71)
        LABEL(70)
        I('PUSH_SCOPE')
        I('MATCH_RANGE', '0', '9')
        I('POP_SCOPE')
        LABEL(71)
        I('RETURN')
        LABEL('STRING')
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', '"')
        I('LIST_START')
        LABEL(74)
        I('BACKTRACK', 75)
        I('PUSH_SCOPE')
        I('BACKTRACK', 73)
        I('MATCH_CHARSEQ', '"')
        I('COMMIT', 72)
        LABEL(72)
        I('FAIL', 'no match expected')
        LABEL(73)
        I('MATCH_ANY')
        I('POP_SCOPE')
        I('LIST_APPEND')
        I('COMMIT', 74)
        LABEL(75)
        I('LIST_END')
        I('BIND', 'xs')
        I('MATCH_CHARSEQ', '"')
        I('ACTION', lambda scope: join(scope['xs'].eval()))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('WS')
        I('PUSH_SCOPE')
        I('LIST_START')
        LABEL(78)
        I('BACKTRACK', 79)
        I('BACKTRACK', 76)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', ' ')
        I('POP_SCOPE')
        I('COMMIT', 77)
        LABEL(76)
        I('PUSH_SCOPE')
        I('MATCH_CHARSEQ', '\n')
        I('POP_SCOPE')
        LABEL(77)
        I('LIST_APPEND')
        I('COMMIT', 78)
        LABEL(79)
        I('LIST_END')
        I('POP_SCOPE')
        I('RETURN')
        LABEL('WB')
        I('PUSH_SCOPE')
        I('BACKTRACK', 81)
        I('CALL', 'NAMECHAR')
        I('COMMIT', 80)
        LABEL(80)
        I('FAIL', 'no match expected')
        LABEL(81)
        I('POP_SCOPE')
        I('RETURN')

class WxCodeGenerator(_Grammar):

    def __init__(self):
        self._instructions = i = []
        self._labels = l = {}
        def I(name, x=None, y=None):
            i.append((name, x, y))
        def LABEL(name):
            l[name] = len(i)
        LABEL('ast')
        I('PUSH_SCOPE')
        I('PUSH_STREAM')
        I('MATCH_CALL_RULE')
        I('BIND', 'x')
        I('POP_STREAM')
        I('ACTION', lambda scope: scope['x'].eval())
        I('POP_SCOPE')
        I('RETURN')
        LABEL('astItems')
        I('BACKTRACK', 2)
        I('PUSH_SCOPE')
        I('CALL', 'ast')
        I('BIND', 'x')
        I('LIST_START')
        LABEL(0)
        I('BACKTRACK', 1)
        I('CALL', 'astItem')
        I('LIST_APPEND')
        I('COMMIT', 0)
        LABEL(1)
        I('LIST_END')
        I('BIND', 'xs')
        I('ACTION', lambda scope: _Builder.create([scope['x'].eval(), scope['xs'].eval()]))
        I('POP_SCOPE')
        I('COMMIT', 3)
        LABEL(2)
        I('PUSH_SCOPE')
        I('ACTION', lambda scope: _Builder.create([]))
        I('POP_SCOPE')
        LABEL(3)
        I('RETURN')
        LABEL('astItem')
        I('PUSH_SCOPE')
        I('CALL', 'ast')
        I('BIND', 'x')
        I('ACTION', lambda scope: _Builder.create([', ', scope['x'].eval()]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('widget')
        I('PUSH_SCOPE')
        I('MATCH_ANY')
        I('BIND', 'name')
        I('CALL', 'ast')
        I('BIND', 'container')
        I('CALL', 'ast')
        I('BIND', 'sizer')
        I('PUSH_STREAM')
        I('LIST_START')
        LABEL(4)
        I('BACKTRACK', 5)
        I('CALL', 'ast')
        I('LIST_APPEND')
        I('COMMIT', 4)
        LABEL(5)
        I('LIST_END')
        I('BIND', 'props')
        I('POP_STREAM')
        I('PUSH_STREAM')
        I('LIST_START')
        LABEL(6)
        I('BACKTRACK', 7)
        I('CALL', 'ast')
        I('LIST_APPEND')
        I('COMMIT', 6)
        LABEL(7)
        I('LIST_END')
        I('BIND', 'inst')
        I('POP_STREAM')
        I('MATCH_ANY')
        I('BIND', 'verbatim')
        I('ACTION', lambda scope: _Builder.create(['class ', scope['name'].eval(), '(', scope['container'].eval(), '):\n\n', _IndentBuilder(), 'def _get_local_props(self):\n', _IndentBuilder(), 'return {\n', _IndentBuilder(), scope['props'].eval(), _DedentBuilder(), '}\n\n', _DedentBuilder(), 'def _create_sizer(self):\n', _IndentBuilder(), 'return ', scope['sizer'].eval(), '\n\n', _DedentBuilder(), 'def _create_widgets(self):\n', _IndentBuilder(), 'pass\n', scope['inst'].eval(), _DedentBuilder(), scope['verbatim'].eval(), _DedentBuilder()]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('instance')
        I('BACKTRACK', 14)
        I('PUSH_SCOPE')
        I('MATCH_STRING', '%space')
        I('CALL', 'ast')
        I('BIND', 'x')
        I('ACTION', lambda scope: _Builder.create(['self._create_space(', scope['x'].eval(), ')\n']))
        I('POP_SCOPE')
        I('COMMIT', 15)
        LABEL(14)
        I('BACKTRACK', 12)
        I('PUSH_SCOPE')
        I('MATCH_ANY')
        I('BIND', 'name')
        I('PUSH_STREAM')
        I('LIST_START')
        LABEL(8)
        I('BACKTRACK', 9)
        I('CALL', 'ast')
        I('LIST_APPEND')
        I('COMMIT', 8)
        LABEL(9)
        I('LIST_END')
        I('BIND', 'xs')
        I('POP_STREAM')
        I('ACTION', lambda scope: _Builder.create(['props = {}\n', 'sizer = {"flag": 0, "border": 0, "proportion": 0}\n', 'handlers = {}\n', scope['xs'].eval(), 'self._create_widget(', scope['name'].eval(), ', props, sizer, handlers)\n']))
        I('POP_SCOPE')
        I('COMMIT', 13)
        LABEL(12)
        I('PUSH_SCOPE')
        I('MATCH_STRING', '%loop')
        I('CALL', 'ast')
        I('BIND', 'x')
        I('LIST_START')
        LABEL(10)
        I('BACKTRACK', 11)
        I('CALL', 'ast')
        I('LIST_APPEND')
        I('COMMIT', 10)
        LABEL(11)
        I('LIST_END')
        I('BIND', 'ys')
        I('ACTION', lambda scope: _Builder.create(['with self._loop():\n', _IndentBuilder(), 'for loopvar in ', scope['x'].eval(), ':\n', _IndentBuilder(), 'pass\n', scope['ys'].eval(), _DedentBuilder(), _DedentBuilder()]))
        I('POP_SCOPE')
        LABEL(13)
        LABEL(15)
        I('RETURN')
        LABEL('layoutAttribute')
        I('BACKTRACK', 20)
        I('PUSH_SCOPE')
        I('MATCH_STRING', 'proportion')
        I('CALL', 'ast')
        I('BIND', 'x')
        I('ACTION', lambda scope: _Builder.create(['sizer["proportion"] = ', scope['x'].eval(), '\n']))
        I('POP_SCOPE')
        I('COMMIT', 21)
        LABEL(20)
        I('BACKTRACK', 18)
        I('PUSH_SCOPE')
        I('MATCH_STRING', 'align')
        I('MATCH_STRING', 'expand')
        I('ACTION', lambda scope: _Builder.create(['sizer["flag"] |= wx.EXPAND\n']))
        I('POP_SCOPE')
        I('COMMIT', 19)
        LABEL(18)
        I('PUSH_SCOPE')
        I('MATCH_STRING', 'margin')
        I('CALL', 'ast')
        I('BIND', 'x')
        I('LIST_START')
        LABEL(16)
        I('BACKTRACK', 17)
        I('CALL', 'ast')
        I('LIST_APPEND')
        I('COMMIT', 16)
        LABEL(17)
        I('LIST_END')
        I('BIND', 'ys')
        I('ACTION', lambda scope: _Builder.create(['sizer["border"] = ', scope['x'].eval(), '\n', scope['ys'].eval()]))
        I('POP_SCOPE')
        LABEL(19)
        LABEL(21)
        I('RETURN')
        LABEL('marginSide')
        I('BACKTRACK', 22)
        I('PUSH_SCOPE')
        I('MATCH_STRING', 'TOP')
        I('ACTION', lambda scope: 'sizer["flag"] |= wx.TOP\n')
        I('POP_SCOPE')
        I('COMMIT', 23)
        LABEL(22)
        I('PUSH_SCOPE')
        I('MATCH_STRING', 'BOTTOM')
        I('ACTION', lambda scope: 'sizer["flag"] |= wx.BOTTOM\n')
        I('POP_SCOPE')
        LABEL(23)
        I('RETURN')
        LABEL('prop')
        I('PUSH_SCOPE')
        I('CALL', 'py')
        I('BIND', 'name')
        I('CALL', 'ast')
        I('BIND', 'default')
        I('ACTION', lambda scope: _Builder.create([scope['name'].eval(), ': ', scope['default'].eval(), ',\n']))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('chain')
        I('PUSH_SCOPE')
        I('CALL', 'ast')
        I('BIND', 'x')
        I('LIST_START')
        LABEL(24)
        I('BACKTRACK', 25)
        I('CALL', 'chainAst')
        I('LIST_APPEND')
        I('COMMIT', 24)
        LABEL(25)
        I('LIST_END')
        I('BIND', 'xs')
        I('ACTION', lambda scope: _Builder.create([scope['x'].eval(), scope['xs'].eval()]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('chainAst')
        I('PUSH_SCOPE')
        I('CALL', 'ast')
        I('BIND', 'x')
        I('ACTION', lambda scope: _Builder.create(['.', scope['x'].eval()]))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('propRef')
        I('PUSH_SCOPE')
        I('CALL', 'py')
        I('BIND', 'name')
        I('ACTION', lambda scope: _Builder.create(['self.prop(', scope['name'].eval(), ')']))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('propAssign')
        I('PUSH_SCOPE')
        I('CALL', 'py')
        I('BIND', 'name')
        I('CALL', 'ast')
        I('BIND', 'value')
        I('ACTION', lambda scope: _Builder.create(['props[', scope['name'].eval(), '] = ', scope['value'].eval(), '\n']))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('propExplode')
        I('PUSH_SCOPE')
        I('CALL', 'py')
        I('BIND', 'name')
        I('ACTION', lambda scope: _Builder.create(['props.update(self.prop(', scope['name'].eval(), '))\n']))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('loopvarRef')
        I('PUSH_SCOPE')
        I('CALL', 'py')
        I('BIND', 'name')
        I('ACTION', lambda scope: _Builder.create(['loopvar[', scope['name'].eval(), ']']))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('loopvarExplode')
        I('BACKTRACK', 26)
        I('PUSH_SCOPE')
        I('MATCH_STRING', '')
        I('ACTION', lambda scope: _Builder.create(['props.update(loopvar)\n']))
        I('POP_SCOPE')
        I('COMMIT', 27)
        LABEL(26)
        I('PUSH_SCOPE')
        I('CALL', 'py')
        I('BIND', 'name')
        I('ACTION', lambda scope: _Builder.create(['props.update(loopvar[', scope['name'].eval(), '])\n']))
        I('POP_SCOPE')
        LABEL(27)
        I('RETURN')
        LABEL('call')
        I('PUSH_SCOPE')
        I('CALL', 'ast')
        I('BIND', 'x')
        I('CALL', 'astItems')
        I('BIND', 'y')
        I('ACTION', lambda scope: _Builder.create([scope['x'].eval(), '(', scope['y'].eval(), ')']))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('handler')
        I('PUSH_SCOPE')
        I('CALL', 'py')
        I('BIND', 'name')
        I('CALL', 'ast')
        I('BIND', 'y')
        I('ACTION', lambda scope: _Builder.create(['handlers[', scope['name'].eval(), '] = lambda event: ', scope['y'].eval(), '\n']))
        I('POP_SCOPE')
        I('RETURN')
        LABEL('layout')
        I('BACKTRACK', 28)
        I('PUSH_SCOPE')
        I('MATCH_STRING', '%layout_rows')
        I('ACTION', lambda scope: 'wx.BoxSizer(wx.VERTICAL)')
        I('POP_SCOPE')
        I('COMMIT', 29)
        LABEL(28)
        I('PUSH_SCOPE')
        I('MATCH_STRING', '%layout_columns')
        I('ACTION', lambda scope: 'wx.BoxSizer(wx.HORIZONTAL)')
        I('POP_SCOPE')
        LABEL(29)
        I('RETURN')
        LABEL('container')
        I('BACKTRACK', 32)
        I('PUSH_SCOPE')
        I('MATCH_STRING', 'frame')
        I('ACTION', lambda scope: 'RLGuiFrame')
        I('POP_SCOPE')
        I('COMMIT', 33)
        LABEL(32)
        I('BACKTRACK', 30)
        I('PUSH_SCOPE')
        I('MATCH_STRING', 'panel')
        I('ACTION', lambda scope: 'RLGuiPanel')
        I('POP_SCOPE')
        I('COMMIT', 31)
        LABEL(30)
        I('PUSH_SCOPE')
        I('MATCH_STRING', 'vscroll')
        I('ACTION', lambda scope: 'RLGuiVScroll')
        I('POP_SCOPE')
        LABEL(31)
        LABEL(33)
        I('RETURN')
        LABEL('string')
        I('PUSH_SCOPE')
        I('CALL', 'py')
        I('POP_SCOPE')
        I('RETURN')
        LABEL('number')
        I('PUSH_SCOPE')
        I('CALL', 'py')
        I('POP_SCOPE')
        I('RETURN')
        LABEL('identifier')
        I('PUSH_SCOPE')
        I('MATCH_ANY')
        I('POP_SCOPE')
        I('RETURN')
        LABEL('py')
        I('PUSH_SCOPE')
        I('MATCH_ANY')
        I('BIND', 'x')
        I('ACTION', lambda scope: repr(scope['x'].eval()))
        I('POP_SCOPE')
        I('RETURN')

def join(items, sep=""):
    return sep.join(items)

def partition(values):
    by_type = defaultdict(list)
    for x in values:
        by_type[x[0]].append(x)
    return by_type

def extract(by_type, name):
    return by_type[name]

if __name__ == "__main__":
    parser = GuiParser()
    codegenerator = WxCodeGenerator()
    try:
        sys.stdout.write(
            codegenerator.run("ast", parser.run("widget", sys.stdin.read()))
        )
    except _MatchError as e:
        sys.exit(e.describe())
