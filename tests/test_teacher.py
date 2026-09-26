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


def test_tetris_objective_takes_a_four_line_clear():
    """Four rows open only in column 10, and a vertical I to fill them."""
    g = Game(seed=0)
    for y in range(H - 4, H):
        g.rows[y] = FULL & ~(1 << 9)
    g.piece = "I"
    turn, col = teacher_move(g, objective="tetris")
    assert (turn, col) == (1, 9)          # turned upright, into the well
    assert g.place(turn, col) == 4


def test_tetris_objective_refuses_a_cheap_single_that_the_flat_one_takes():
    """One row needs a single cell in column 1; the alternative keeps the well ready."""
    g = Game(seed=0)
    g.rows[H - 1] = FULL & ~0b1
    for y in range(H - 4, H - 1):
        g.rows[y] = FULL & ~(1 << 9) & ~0b1
    g.piece = "I"
    flat = teacher_move(g, objective="eltetris")
    keen = teacher_move(g, objective="tetris")
    assert flat != keen, "the two objectives should disagree here"


def test_tetris_objective_still_clears_when_the_stack_is_high():
    """Above STACK_PANIC it stops holding out for a Tetris."""
    from tetris.teacher import STACK_PANIC
    g = Game(seed=0)
    for y in range(H - STACK_PANIC - 2, H):
        g.rows[y] = FULL & ~0b1
    g.piece = "I"
    turn, col = teacher_move(g, objective="tetris")
    assert g.place(turn, col) > 0


def test_two_ply_matches_one_ply_when_it_cannot_improve():
    from tetris.lookahead import all_scores_n
    g = Game(seed=3)
    one = all_scores_n(g, 1, "eltetris")
    two = all_scores_n(g, 2, "eltetris")
    assert set(one) == set(two)
    assert all(v > -1e6 for v in two.values())
