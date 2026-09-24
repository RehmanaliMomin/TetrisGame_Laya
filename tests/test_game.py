from tetris.game import FULL, H, SHAPES, W, Game, column_heights, column_holes, drop, shape, underside, width


def test_distinct_rotations():
    assert {p: len(s) for p, s in SHAPES.items()} == {"I": 2, "O": 1, "T": 4, "S": 2, "Z": 2, "J": 4, "L": 4}
    assert shape("I", 2) == shape("I", 0) and shape("O", 3) == shape("O", 0)


def test_shape_helpers():
    t_flip = shape("T", 2)  # point down
    assert width(t_flip) == 3 and underside(t_flip) == [1, 0, 1]
    assert underside(shape("T", 0)) == [0, 0, 0]
    assert width(shape("I", 1)) == 1


def test_drop_lands_on_stack():
    rows = [0] * H
    rows[H - 1] = 0b1  # one block bottom-left
    new, cells, cleared, topped = drop(rows, shape("I", 1), 0)
    assert not topped and not cleared
    assert sorted(y for _, y in cells) == [H - 5, H - 4, H - 3, H - 2]
    assert column_heights(new)[0] == 5


def test_line_clear_and_score():
    g = Game(seed=1)
    g.rows[H - 1] = FULL & ~0b1111  # bottom row missing the 4 leftmost cells
    g.colors[H - 1] = ["."] * 4 + ["Z"] * 6
    g.piece = "I"
    assert g.place(0, 0) == 1
    assert g.rows[H - 1] == 0 and g.lines == 1 and g.score == 100 and g.clears[1] == 1
    assert all(c == "." for c in g.colors[H - 1])


def test_holes():
    rows = [0] * H
    rows[H - 2] = 0b10
    assert column_holes(rows) == [0, 1] + [0] * (W - 2)


def test_top_out_ends_game():
    g = Game(seed=2)
    for y in range(1, H):
        g.rows[y] = 0b1
    g.piece = "I"
    assert g.tops_out(1, 0)
    g.place(1, 0)
    assert g.done and g.end_cause == "topped out"


def test_invalid_column_rejected():
    g = Game(seed=3)
    g.piece = "I"
    assert not g.is_valid(0, W - 3)
    try:
        g.place(0, W - 3)
        assert False, "should raise"
    except ValueError:
        pass


def test_seeded_bag_is_deterministic_and_fair():
    a, b = Game(seed=7), Game(seed=7)
    seq_a = [a.piece, a.next] + [a._draw() for _ in range(12)]
    seq_b = [b.piece, b.next] + [b._draw() for _ in range(12)]
    assert seq_a == seq_b
    assert sorted(seq_a[:7]) == sorted("IOTSZJL")
