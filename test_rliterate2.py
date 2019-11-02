from unittest.mock import Mock, call

import rliterate2

def test_notification():
    listeners = Mock()
    immutable = rliterate2.Immutable({
        "foo": 1,
        "bar": {
            "foobar": 2,
        },
    })
    immutable.listen(listeners.foo, ["foo"])
    immutable.listen(listeners.bar, ["bar"])
    immutable.listen(listeners.foobar, ["bar", "foobar"])

    listeners.reset_mock()
    immutable.replace(["foo"], 2)
    assert listeners.mock_calls == [
        call.foo(),
    ]

    listeners.reset_mock()
    immutable.replace(["bar", "foobar"], 3)
    assert listeners.mock_calls == [
        call.bar(),
        call.foobar(),
    ]

    listeners.reset_mock()
    immutable.replace(["bar"], {"foobar", 3})
    assert listeners.mock_calls == [
        call.bar(),
        call.foobar(),
    ]
