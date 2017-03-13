# basic search techniques
# written by Aleksander Sadikov, November 2007 (and updated in November 2008 and March 2010)
# for AI practical classes @ University of Ljubljana

import bisect

INFINITY = 1000000


def df(pos, depth):
    # depth-first search
    if depth <= 0:
        return None

    if pos.solved():
        return []

    for move in pos.generate_moves():
        pos.move(move)
        found_path = df(pos, depth - 1)
        pos.undo_move(move)

        if found_path is not None:
            found_path.insert(0, move)
            return found_path

    return None


def df_basic(pos):
    # depth-first search
    if pos.solved():
        return []

    for move in pos.generate_moves():
        pos.move(move)
        found_path = df_basic(pos)
        pos.undo_move(move)

        if found_path is not None:
            found_path.insert(0, move)
            return found_path

    return None


def id(pos):
    # iterative deepening
    depth = 0
    while True:
        path = df(pos, depth)
        if path is not None:
            return path

        depth += 1
        print depth


def bf(pos):
    # breadth-first search
    cand_paths = [[]]
    while cand_paths:
        path = aux_bf(pos, cand_paths)
        if path is not None:
            return path

    return None  # no more candidate paths and solution was not found


def aux_bf(pos, cand_paths):
    # auxiliary routine for expanding nodes in breadth-first search
    path = cand_paths.pop(0)
    pos.execute_sequence(path)
    if pos.solved():
        pos.unto_sequence(path)
        return path
    for move in pos.generate_moves():
        cand_paths.append(path + [move])
    pos.unto_sequence(path)
    return None


def A_star(pos, limit=25000):
    # A* best-first heuristic search
    # warning: only for educational purposes as it shows how similar it really is to BF
    # path_prior: priority queue (holds f-values of candidate paths in sorted order)
    # cand_paths: list of candidate paths; sorted according to order in path_prior
    # cand_ids: list of IDs of nodes reached by candidate paths; needed to detect transpositions (also ordered)
    # visited: dictionary of already generated nodes and best g-values seen to reach them (thus no duplicates!)
    path_prior, cand_paths, cand_ids = [pos.evaluate()], [[]], [pos.id()]
    visited = {pos.id(): 0}
    ix = 1
    bh = pos.evaluate()
    while cand_paths:
        if ix > limit:
            return None  # limit on expanded nodes reached

        ix += 1
        # for testing
        pos.execute_sequence(cand_paths[0])
        btemp = pos.evaluate()
        if btemp < bh:
            bh = btemp

        pos.unto_sequence(cand_paths[0])
        path = aux_A_star(pos, (path_prior, cand_paths, cand_ids, visited))
        if path is not None:
            return path

    return None  # no more candidate paths and solution was not found


def aux_A_star(pos, (path_prior, cand_paths, cand_ids, visited)):
    # auxiliary routine for expanding nodes in A* best-first heuristic search
    # the first is taken, but actually it should be randomly chosen between the equal best candidates
    path = cand_paths.pop(0)
    pos_id = cand_ids.pop(0)
    path_prior.pop(0)
    pos.execute_sequence(path)
    if pos.solved():
        pos.unto_sequence(path)
        return path
    g = visited[pos_id]  # get cost of current path (g)
    for move, cost in pos.generate_moves():
        pos.move((move, cost))
        add_cand(pos, g + cost, path + [(move, cost)], (path_prior, cand_paths, cand_ids, visited))
        pos.undo_move((move, cost))
    pos.unto_sequence(path)
    return None


def add_cand(pos, g, path, (path_prior, cand_paths, cand_ids, visited)):
    pos_id = pos.id()
    if pos_id in visited:
        # this node was reached previously (transposition found)
        if g >= visited[pos_id]:
            # new path to this node worse (or equal) than the best previously found: ignore it
            return
            # #        else:
            # #            # new path better: delete the previous entry (new entry will be added later)
            # #            try:
            # #                ix = CandIDs.index(PosID)       # this is unfortunately slow
            # #                CandIDs.pop(ix) ; CandPaths.pop(ix) ; PathPrior.pop(ix)
            # #            except:
            # #                pass

    # add new entry
    f = g + pos.evaluate()
    ix = bisect.bisect_left(path_prior, f)
    cand_ids.insert(ix, pos_id)
    cand_paths.insert(ix, path)
    path_prior.insert(ix, f)
    visited[pos_id] = g


def ida(pos):
    # iterative deepening A*
    fbound = pos.evaluate()
    while True:
        res = r_ida(pos, 0, fbound)
        if isinstance(res, list):
            return res

        fbound = res


def r_ida(pos, g, fbound):
    # recursive depth-first search with F-bound for use with IDA*
    # if goal is found it returns path-to-goal (as a list of moves)
    # else it returns a new fbound candidate (as a number; this is at the same time a signal of failure)
    f = g + pos.evaluate()
    if f > fbound:
        return f

    if pos.solved():
        return []

    newfbound = INFINITY
    for move, cost in pos.generate_moves():
        pos.move((move, cost))
        res = r_ida(pos, g + cost, fbound)
        pos.undo_move((move, cost))
        if isinstance(res, list):
            res.insert(0, (move, cost))
            return res
        else:
            if res < newfbound:
                newfbound = res
    return newfbound


def rbfs(pos):
    # recursive best-first search (top level routine; it starts the recursive chain)
    return r_rbfs(pos, 0, pos.evaluate(), INFINITY)


def r_rbfs(pos, g, f, bound):
    # recursive best-first search algorithm (RBFS)
    # if goal is found it returns path-to-goal (as a list of moves)
    # else it returns a new bound (as a number; this is at the same time a signal of failure)
    f = g + pos.evaluate()
    if f > bound:
        return f
    if pos.solved():
        return []

    fs = [(INFINITY, None)]  # acts as a guard for insertion below and solves single-child case
    for move, cost in pos.generate_moves():
        pos.move((move, cost))
        newf = g + cost + pos.evaluate()
        if f < f:
            newf = max(f, newf)
        pos.undo_move((move, cost))
        ins_move_by_f_value(newf, (move, cost), fs)

    if fs == [(INFINITY, None)]:
        return INFINITY  # no legal moves in this position

    while fs[0][0] <= bound and fs[0][0] < INFINITY:
        f1, (move, cost) = fs.pop(0)
        pos.move((move, cost))
        res = r_rbfs(pos, g + cost, f1, min(bound, fs[0][0]))  # after pop(), fs[0] is the second candidate move!
        pos.undo_move((move, cost))
        if isinstance(res, list):
            return [(move, cost)] + res  # solution found; add current move to it and return
        ins_move_by_f_value(res, (move, cost), fs)
    return fs[0][0]


def ins_move_by_f_value(newf, move, fs):
    # insert the pair (f-value, move) into a sorted list
    # NOTE: move with f-value of INFINITY is not inserted at all (does not change the behaviour of the algorithm!)
    for ix, (cf, cmove) in enumerate(fs):
        if newf < cf:
            fs.insert(ix, (newf, move))
            break


def rbfs_verbose(pos):
    # recursive best-first search (top level routine; it starts the recursive chain)
    return vr_rbfs(pos, 0, pos.evaluate(), INFINITY, 0)


def vr_rbfs(pos, g, f, bound, depth):
    # recursive best-first search algorithm (RBFS)
    # if goal is found it returns path-to-goal (as a list of moves)
    # else it returns a new bound (as a number; this is at the same time a signal of failure)
    print depth * "  ", "Enter (Pos, g, F, bound):", pos.State, g, f, bound
    f = g + pos.evaluate()
    if f > bound:
        print depth * "  ", "Exceeded bound (f, bound):", f, bound
        return f
    if pos.solved():
        print depth * "  ", "Solution found! (res):", []
        return []

    fs = [(INFINITY, None)]  # acts as a guard for insertion below and solves single-child case
    for move, cost in pos.generate_moves():
        pos.move((move, cost))
        newf = g + cost + pos.evaluate()
        if f < f:
            newf = max(f, newf)
        pos.undo_move((move, cost))
        ins_move_by_f_value(newf, (move, cost), fs)

    if fs == [(INFINITY, None)]:
        return INFINITY  # no legal moves in this position
    while fs[0][0] <= bound and fs[0][0] < INFINITY:
        print depth * "  ", "Before call (g, fs):", g, fs[:-1]
        f1, (move, cost) = fs.pop(0)
        pos.move((move, cost))
        res = vr_rbfs(pos, g + cost, f1, min(bound, fs[0][0]), depth + 1)
        # after pop(), fs[0] is the second candidate move!
        ins_move_by_f_value(res, (move, cost), fs)
        pos.undo_move((move, cost))
        if isinstance(res, list):  # solution found; add current move to it and return
            res.insert(0, (move, cost))
            print depth * "  ", "Returning solution (res):", res
            return res
    print depth * "  ", "Before returning (g, fs):", g, fs[:-1]
    print depth * "  ", "Returning new F-value (f1):", fs[0][0]
    return fs[0][0]








