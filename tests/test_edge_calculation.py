from models import calculate_edge_ev


def test_edge_and_ev_for_up_side() -> None:
    ev_up, ev_down, edge_up, edge_down = calculate_edge_ev(
        p_up=0.72,
        price_up=0.60,
        price_down=0.41,
        fees=0.0,
        slippage=0.01,
    )
    assert round(edge_up, 4) == 0.12
    assert round(edge_down, 4) == -0.13
    assert ev_up > 0
    assert ev_down < 0


def test_edge_and_ev_include_costs() -> None:
    ev_up, _, edge_up, _ = calculate_edge_ev(
        p_up=0.60,
        price_up=0.58,
        price_down=0.43,
        fees=0.01,
        slippage=0.02,
    )
    assert round(edge_up, 4) == 0.02
    assert ev_up < edge_up
