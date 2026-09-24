from tetris.game import FULL, H, Game
from tetris.policy import RandomPolicy, TeacherPolicy
from tetris.teacher import all_scores, board_features, column_target, teacher_move, turn_target


def play(policy, seed, pieces):
    g = Game(seed=seed, max_pieces=pieces)
    while not g.done:
        d = policy.decide(g)
        g.place(d["turn"], d["col"])
    return g


def test_empty_board_features():
    f = board_features([0] * H)
    assert f["holes"] == 0 and f["wells"] == 0
    assert f["row_transitions"] == 2 * H  # wall -> empty -> wall on every row
    assert f["col_transitions"] == 10     # every column: empty -> floor


def test_teacher_takes_the_line_clear():
    g = Game(seed=0)
    g.rows[H - 1] = FULL & ~0b1111
    g.piece = "I"
    assert teacher_move(g) == (0, 0)


def test_targets_are_distributions():
    g = Game(seed=5)
    g.piece = "I"
    t = turn_target(g)
    assert abs(sum(t) - 1) < 1e-9 and t[2] == t[3] == 0.0  # I has 2 distinct turns
    c = column_target(g, 0)
    assert abs(sum(c) - 1) < 1e-9 and c[7:] == [0.0, 0.0, 0.0]  # horizontal I fits columns 0..6


def test_teacher_beats_random():
    teacher = play(TeacherPolicy(), 11, 300)
    rand = play(RandomPolicy(1), 11, 300)
    assert teacher.end_cause == "piece limit" and teacher.lines >= 100
    assert rand.lines < teacher.lines / 5


def test_every_scored_placement_is_valid():
    g = Game(seed=4)
    for (t, c) in all_scores(g):
        assert g.is_valid(t, c)
