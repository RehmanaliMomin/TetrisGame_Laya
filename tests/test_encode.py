from tetris.encode import COLUMN_KEYS, TURN_KEYS, encode_column_state, encode_turn_state
from tetris.game import H, Game


def test_turn_state_text():
    g = Game(seed=0)
    g.piece = "T"
    g.rows[H - 1] = 0b11
    g.rows[H - 2] = 0b01
    s = encode_turn_state(g)
    assert s == ("piece: T | heights: 2 1 0 0 0 0 0 0 0 0 | steps: -1 -1 0 0 0 0 0 0 0 "
                 "| holes: 0 0 0 0 0 0 0 0 0 0 | max height: 2")


def test_column_state_mentions_turned_shape():
    g = Game(seed=0)
    g.piece = "T"
    s = encode_column_state(g, 2)
    assert s.startswith("piece: T | turn: flip | shape: width 3, underside 1 0 1 | heights: 0 0")
    assert encode_column_state(g, 2) == s  # deterministic


def test_option_keys():
    assert TURN_KEYS == ["spawn", "right", "flip", "left"]
    assert COLUMN_KEYS == ["c%d" % i for i in range(1, 11)]
